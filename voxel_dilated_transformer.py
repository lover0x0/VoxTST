import itertools
from typing import Optional, Sequence, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from einops import rearrange
from flash_attention import flash_attn_func
from dec import DEC


def window_partition(x, window_size):
    """
    Args:
        x:  (B, D, H, W, C)
        window_size: (window_size_D, window_size_H, window_size_W)
    Returns:
        windows: (num_windows * B, D_w, H_w, W_w, C)
    """
    B, D, H, W, C = x.shape
    D_w, H_w, W_w = window_size

    x = x.view(B,
               D // D_w, D_w,
               H // H_w, H_w,
               W // W_w, W_w,
               C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, D_w, H_w, W_w, C)
    return windows

def window_reverse(windows, window_size, dims):
    """
    Args:
        windows: (num_windows * B, num_heads, D_w, H_w, W_w, C)
        window_size: (D_w, H_w, W_w)
        dims: (B, D, H, W)
    Returns:
        x: (B, num_heads, D, H, W, C)
    """
    B, D, H, W = dims
    D_w, H_w, W_w = window_size
    num_heads = windows.shape[1]
    x = windows.view(B,
                     D // D_w, H // H_w, W // W_w,
                     num_heads,
                     D_w, H_w, W_w,
                     -1)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7, 8).contiguous()
    x = x.view(B, num_heads, D, H, W, -1)
    return x

class DilatedAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, dilation_rates, window_sizes, attn_drop, dropout=0.0):
        super(DilatedAttention3D, self).__init__()
        assert len(dilation_rates) == len(window_sizes), "dilation_rates and window_sizes must have the same length"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dilation_rates = dilation_rates
        self.window_sizes = window_sizes
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = attn_drop

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def dense_to_sparse(self, x, dilation_rate):
        """
        x: (batch_size, num_heads, D, H, W, head_dim)
        """
        batch_size, num_heads, D, H, W, head_dim = x.shape
        r = dilation_rate
        r_total = r ** 3

        assert num_heads % r_total == 0, "num_heads must be divisible by dilation_rate^3"
        num_head_groups = num_heads // r_total

        pad_D = (r - D % r) % r
        pad_H = (r - H % r) % r
        pad_W = (r - W % r) % r

        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            x = F.pad(x, (0, 0, 0, pad_W, 0, pad_H, 0, pad_D))

        D_padded, H_padded, W_padded = x.shape[2], x.shape[3], x.shape[4]

        x = x.view(batch_size, r_total, num_head_groups, D_padded // r, r, H_padded // r, r, W_padded // r, r, head_dim)

        x = x.permute(0, 2, 4, 6, 8, 1, 3, 5, 7, 9).contiguous() #(batch_size, num_head_groups, r, r, r, r_total, D', H', W', head_dim)

        idx = torch.arange(r, device=x.device)
        idx_grid = torch.stack(torch.meshgrid(idx, idx, idx, indexing='ij'), -1).view(-1, 3)  # (r_total, 3)

        x_list = []
        for i in range(r_total):
            offset = idx_grid[i]
            x_i = x[:, :, offset[0], offset[1], offset[2], i, :, :, :, :]# (batch_size, num_head_groups, D', H', W', head_dim)
            x_list.append(x_i)

        x = torch.stack(x_list, dim=1)  # (batch_size, r_total, num_head_groups, D', H', W', head_dim)
        x = x.view(batch_size, num_heads, D_padded // r, H_padded // r, W_padded // r, head_dim)

        x = x.view(batch_size, num_heads, -1, head_dim).permute(0, 2, 1, 3)  # (batch_size, N, num_heads, head_dim)

        return x, (D_padded, H_padded, W_padded)

    def sparse_to_dense(self, x, lse, original_shape, dilation_rate):
        """
        x: (batch_size, N, num_heads, head_dim)
        lse: (batch_size, num_heads, N)
        """
        batch_size, N, num_heads, head_dim = x.shape
        D, H, W = original_shape
        r = dilation_rate
        r_total = r ** 3
        num_head_groups = num_heads // r_total

        D_padded = D + (r - D % r) % r
        H_padded = H + (r - H % r) % r
        W_padded = W + (r - W % r) % r

        D_s = D_padded // r
        H_s = H_padded // r
        W_s = W_padded // r

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, num_heads, D_s, H_s, W_s, head_dim)
        lse = lse.view(batch_size, num_heads, D_s, H_s, W_s, 1)

        x = x.view(batch_size, r_total, num_head_groups, D_s, H_s, W_s, head_dim)
        lse = lse.view(batch_size, r_total, num_head_groups, D_s, H_s, W_s, 1)

        out = torch.zeros(batch_size, num_head_groups, r, r, r, r_total, D_s, H_s, W_s, head_dim, device=x.device,
                          dtype=x.dtype)
        lse_out = torch.zeros(batch_size, num_head_groups, r, r, r, r_total, D_s, H_s, W_s, 1, device=lse.device,
                              dtype=lse.dtype)

        idx = torch.arange(r, device=x.device)
        idx_grid = torch.stack(torch.meshgrid(idx, idx, idx, indexing='ij'), -1).view(-1, 3)  # (r_total, 3)

        for i in range(r_total):
            offset = idx_grid[i]
            out[:, :, offset[0], offset[1], offset[2], i, :, :, :, :] = x[:, i, :, :, :, :]
            lse_out[:, :, offset[0], offset[1], offset[2], i, :, :, :, :] = lse[:, i, :, :, :, :]

        out = out.permute(0, 5, 1, 6, 2, 7, 3, 8, 4, 9).contiguous()
        out = out.view(batch_size, num_heads, D_padded, H_padded, W_padded, head_dim)
        # (batch_size, num_heads, D_padded, H_padded, W_padded, head_dim)
        lse_out = lse_out.permute(0, 5, 1, 6, 2, 7, 3, 8, 4, 9).contiguous()
        lse_out = lse_out.view(batch_size, num_heads, D_padded, H_padded, W_padded, 1)
        # (batch_size, num_heads, D_padded, H_padded, W_padded, 1)

        out = out[:, :, :D, :H, :W, :]
        lse_out = lse_out[:, :, :D, :H, :W, :]
        # print('lse_out',lse_out.shape)
        lse_out = lse_out.masked_fill_(lse_out == 0, -1e8)

        return out, lse_out

    def forward(self, x, mask, is_causal=False):
        """
        x: (batch_size, D, H, W, embed_dim)
        """
        B, D, H, W, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        all_outs = []
        all_lses = []

        for idx, (window_size, dilation_rate) in enumerate(zip(self.window_sizes, self.dilation_rates)):

            # print('window_size', window_size)
            # print('dilation_rate', dilation_rate)
            q_windows = window_partition(q, window_size)  # (num_windows * B, D_w, H_w, W_w, C)
            k_windows = window_partition(k, window_size)
            v_windows = window_partition(v, window_size)
            mask_windows = window_partition(mask, window_size)  # (num_windows * B, D_w, H_w, W_w, 1)
            # print('mask_windows: ', mask_windows.shape)

            q_windows = rearrange(q_windows, 'b D H W (h c) -> b h D H W c', h=self.num_heads)
            k_windows = rearrange(k_windows, 'b D H W (h c) -> b h D H W c', h=self.num_heads)
            v_windows = rearrange(v_windows, 'b D H W (h c) -> b h D H W c', h=self.num_heads)
            mask_windows = mask_windows.unsqueeze(1)  # (b, 1, D_w, H_w, W_w, 1)
            mask_windows = mask_windows.expand(-1, self.num_heads, -1, -1, -1, -1)

            # print('mask_windows_repeat: ', mask_windows.shape)

            q_sparse, (D_padded, H_padded, W_padded) = self.dense_to_sparse(q_windows, dilation_rate)  # (batch_size * num_heads, N, head_dim)
            k_sparse, _ = self.dense_to_sparse(k_windows, dilation_rate)
            v_sparse, _ = self.dense_to_sparse(v_windows, dilation_rate)
            mask_sparse, _ = self.dense_to_sparse(mask_windows, dilation_rate)  # (batch_size, N, num_heads, 1)
            # print('mask_sparse: ', mask_sparse.shape)
            attn_mask_sparse = mask_sparse.squeeze(-1).permute(0, 2, 1) # (B, num_heads, N)
            attn_mask_sparse = attn_mask_sparse.unsqueeze(2).expand(-1, -1, attn_mask_sparse.shape[2], -1)
            # print('attn_mask_sparse', attn_mask_sparse.shape)
            attn_mask_sparse = attn_mask_sparse.masked_fill(attn_mask_sparse == 0, -1e9)
            attn_mask_sparse = attn_mask_sparse.masked_fill(attn_mask_sparse == 1, 0.0)


            q_sparse = q_sparse.to(torch.float16)  # (batch_size, N, num_heads, head_dim)
            k_sparse = k_sparse.to(torch.float16)
            v_sparse = v_sparse.to(torch.float16)
            attn_mask_sparse = attn_mask_sparse.to(torch.float16)

            attn_output_sparse, lse = flash_attn_func(q_sparse, k_sparse, v_sparse, self.attn_drop, attn_mask_sparse, None, is_causal)
            # print('attn_output_sparse', attn_output_sparse)
            # print('lse', lse)
            # attn_output_sparse: (batch_size, N, num_heads, head_dim)
            # lse: (batch_size, num_heads, N)
            lse = torch.where(torch.isinf(lse), torch.full_like(lse, -1e9), lse)

            original_shape = (D_padded, H_padded, W_padded)  # (D_w, H_w, W_w)
            attn_output_dense, lse_dense = self.sparse_to_dense(
                attn_output_sparse,
                lse,
                original_shape,
                dilation_rate
            )
            # attn_output_dense: (batch_size, num_heads, D_w, H_w, W_w, head_dim)
            # lse_dense: (batch_size, num_heads, D_w, H_w, W_w, 1)

            attn_output = window_reverse(
                attn_output_dense,
                window_size,
                (B, D, H, W)
            )  # (B, num_heads, D, H, W, head_dim)
            lse_output = window_reverse(
                lse_dense,
                window_size,
                (B, D, H, W)
            )  # (B, num_heads, D, H, W, 1)

            mask_expended = mask.unsqueeze(1)  # (b, 1, D_w, H_w, W_w, 1)
            mask_expanded = mask_expended.expand(-1, self.num_heads, -1, -1, -1, -1)  # (B, num_heads, D, H, W, 1)
            # print('attn_output', attn_output.shape)
            # print('mask_expanded', mask_expanded.shape)
            attn_output = attn_output * mask_expanded

            o = attn_output.view(B * self.num_heads, D * H * W, self.head_dim)
            lse = lse_output.view(B * self.num_heads, D * H * W, 1)
            all_outs.append(o)
            all_lses.append(lse)

        with torch.no_grad():
            max_lse = torch.stack(all_lses, dim=0).max(0)[0]  # (B * num_heads, D * H * W, 1)
            all_lses = [torch.exp(lse - max_lse) for lse in all_lses]
            lse_sum = torch.stack(all_lses, dim=0).sum(0)
            all_lses = [lse / lse_sum for lse in all_lses]

        out = torch.stack([o * w.type_as(o) for o, w in zip(all_outs,all_lses)], dim=0).sum(dim=0)  # (B * num_heads, D * H * W, head_dim)
        out = rearrange(out, '(b h) l d -> b l (h d)', h=self.num_heads)

        out = out.view(B, D, H, W, self.embed_dim)

        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class Voxel_DilatedTransformerBlock3D(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[tuple],
        dilation_rates: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            dilation_ratio:dilation_ratio.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = DilatedAttention3D(
            dim,
            num_heads,
            dilation_rates,
            window_size,
            attn_drop=attn_drop,
            dropout=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop)

    def forward_part1(self, x, mask):
        b, d, h, w, c = x.shape
        x = self.norm1(x)
        x = self.attn(x, mask=mask)

        return x, mask

    def forward_part2(self, x):
        x = self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x, mask):
        shortcut = x
        if self.use_checkpoint:
            x, mask = checkpoint.checkpoint(self.forward_part1, x, mask)
        else:
            x, mask = self.forward_part1(x, mask)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x, mask



class Voxel_Merging(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, c_multiplier: int = 2
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
        """

        super().__init__()
        self.dim = dim

        # Skip dimension reduction on the temporal dimension

        self.reduction = nn.Linear(8 * dim, c_multiplier * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, mask):
        gray_matter_mask = mask.expand(-1, -1, -1, -1, x.size(-1))  # [b, d, h, w, c]
        if not gray_matter_mask.is_contiguous():
            gray_matter_mask = gray_matter_mask.contiguous()

        # 将非灰质体素的特征置为 0
        x = x * gray_matter_mask  # 在通道维度上进行逐元素相乘，非灰质体素特征将变为 0
        x_shape = x.size()
        b, d, h, w, c = x_shape
        x_list = []
        mask_list = []
        for i, j, k in itertools.product(range(2), range(2), range(2)):
            x_i = x[:, i::2, j::2, k::2, :]
            x_list.append(x_i)
            mask_i = mask[:, i::2, j::2, k::2, :]
            mask_list.append(mask_i)
        x = torch.cat(x_list, -1)
        mask = torch.cat(mask_list, -1)
        mask = mask.sum(dim=-1) > 0  # [B, D//2, H//2, W//2]

        x = self.norm(x)
        x = self.reduction(x)

        return x, mask


MERGING_MODE = {"Voxel_Merging": Voxel_Merging}


class BasicLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[tuple],
        dilation_rates: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        c_multiplier: int = 2,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                Voxel_DilatedTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    dilation_rates=dilation_rates,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, c_multiplier=c_multiplier
            )

    def forward(self, x, mask):
        if mask is None:
            b, c, d, h, w = x.size()
            mask = torch.ones(b, d, h, w, device=x.device)
        b, c, d, h, w = x.size()
        x = rearrange(x, "b c d h w -> b d h w c")
        # print('mask_before', mask.shape)
        # print('mask_min_value', mask.min())

        if mask is not None:
            if mask.dim() == 4:
                mask = rearrange(mask, "b d h w -> b d h w 1")
            else:
                mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, D, H, W, 1]
                mask = mask.expand(b, -1, -1, -1, -1)

        # print('x', x.shape)
        # print('mask', mask.shape)

        for blk in self.blocks:
            x, mask = blk(x, mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x, mask = self.downsample(x, mask)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x, mask


# Basic layer for full attention
class BasicLayer_FullAttention(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
            dim: int,
            depth: int,
            num_heads: int,
            window_size: Sequence[tuple],
            dilation_rates: Sequence[int],
            drop_path: list,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            norm_layer: Type[LayerNorm] = nn.LayerNorm,
            c_multiplier: int = 2,
            downsample: Optional[nn.Module] = None,
            use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                Voxel_DilatedTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    dilation_rates=dilation_rates,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                dim=dim, norm_layer=norm_layer, c_multiplier=c_multiplier
            )

    def forward(self, x, mask):
        if mask is None:
            b, c, d, h, w = x.size()
            mask = torch.ones(b, d, h, w, device=x.device)
        b, c, d, h, w = x.size()
        window_size = self.window_size
        x = rearrange(x, "b c d h w -> b d h w c")

        if mask is not None:
            if mask.dim() == 4:
                mask = rearrange(mask, "b d h w -> b d h w 1")
            else:
                mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, D, H, W, 1]
                mask = mask.expand(b, -1, -1, -1, -1)

        for blk in self.blocks:
            x, mask = blk(x, mask)
        x = x.view(b, d, h, w, -1)
        if self.downsample is not None:
            x, mask = self.downsample(x, mask)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x, mask


class PositionalEmbedding(nn.Module):
    """
    Absolute positional embedding module
    """

    def __init__(
        self, dim: int, voxel_dim: tuple
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            patch_num: total number of patches per time frame
            time_num: total number of time frames
        """

        super().__init__()
        self.dim = dim
        self.voxel_dim = voxel_dim
        d, h, w = voxel_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, d, h, w))

        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, c, d, h, w = x.shape

        x = x + self.pos_embed

        return x


class Voxel_DilatedTransformer3D(nn.Module):


    def __init__(
        self,
        img_size: Tuple,
        embed_dim: int,
        window_size_layer1: Sequence[tuple],
        window_size_layer2: Sequence[tuple],
        window_size_layer3: Sequence[tuple],
        dilation_rates_layer1: Sequence[int],
        dilation_rates_layer2: Sequence[int],
        dilation_rates_layer3: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = True,
        Third_Layer_full_MSA: bool = True,
        downsample="Voxel_Merging",
        num_classes=2,
        to_float: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            downsample: module used for downsampling, available options are `"Voxel_Merging"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).


            c_multiplier: multiplier for the feature length after patch merging
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size_layer1 = window_size_layer1
        self.window_size_layer2 = window_size_layer2
        self.window_size_layer3 = window_size_layer3
        self.dilation_rates_layer1 = dilation_rates_layer1
        self.dilation_rates_layer2 = dilation_rates_layer2
        self.dilation_rates_layer3 = dilation_rates_layer3
        self.to_float = to_float
        grid_size = img_size
        self.grid_size = grid_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        voxel_dim = (img_size[0], img_size[1], img_size[2])

        #print img, patch size, patch dim
        print("img_size: ", img_size)
        print("voxel_dim: ", voxel_dim)
        self.pos_embeds = nn.ModuleList()
        pos_embed_dim = embed_dim
        for i in range(self.num_layers):
            self.pos_embeds.append(PositionalEmbedding(pos_embed_dim, voxel_dim))
            pos_embed_dim = pos_embed_dim * c_multiplier
            voxel_dim = (voxel_dim[0]//2, voxel_dim[1]//2, voxel_dim[2]//2)

        # build layer
        self.layers = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
    
        layer = BasicLayer(
            dim=int(embed_dim),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=self.window_size_layer1,
            dilation_rates=self.dilation_rates_layer1,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
            downsample=down_sample_mod if 0 < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint,
        )
        self.layers.append(layer)

        # exclude last layer
        layer = BasicLayer(
            dim=int(embed_dim * (c_multiplier**1)),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=self.window_size_layer2,
            dilation_rates=self.dilation_rates_layer2,
            drop_path=dpr[sum(depths[:1]): sum(depths[: 1 + 1])],
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            c_multiplier=c_multiplier,
            downsample=down_sample_mod if 1 < self.num_layers - 1 else None,
            use_checkpoint=use_checkpoint,
        )
        self.layers.append(layer)

        if not Third_Layer_full_MSA:
            layer = BasicLayer(
                dim=int(embed_dim * (c_multiplier ** 2)),
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=self.window_size_layer3,
                dilation_rates=self.dilation_rates_layer3,
                drop_path=dpr[sum(depths[:2]): sum(depths[: 2 + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=down_sample_mod if 2 < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        else:
            #################Full MSA for Third layer#####################

            self.third_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 2)),
                self.grid_size[1] // int(2 ** (self.num_layers - 2)),
                self.grid_size[2] // int(2 ** (self.num_layers - 2)),
            )

            layer = BasicLayer_FullAttention(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 2)),
                depth=depths[(self.num_layers - 2)],
                num_heads=num_heads[(self.num_layers - 2)],
                # change the window size to the entire grid size
                window_size=[self.third_window_size],
                dilation_rates=[1],
                drop_path=dpr[sum(depths[: (self.num_layers - 2)]) : sum(depths[: (self.num_layers - 2) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=down_sample_mod if 2 < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        if not last_layer_full_MSA:
            layer = BasicLayer(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                window_size=self.window_size_layer3,
                dilation_rates=self.dilation_rates_layer3,
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        else:
            #################Full MSA for last layer#####################

            self.last_window_size = (
                self.grid_size[0] // int(2 ** (self.num_layers - 1)),
                self.grid_size[1] // int(2 ** (self.num_layers - 1)),
                self.grid_size[2] // int(2 ** (self.num_layers - 1)),
            )

            layer = BasicLayer_FullAttention(
                dim=int(embed_dim * c_multiplier ** (self.num_layers - 1)),
                depth=depths[(self.num_layers - 1)],
                num_heads=num_heads[(self.num_layers - 1)],
                # change the window size to the entire grid size
                window_size=[self.last_window_size],
                dilation_rates=[1],
                drop_path=dpr[sum(depths[: (self.num_layers - 1)]) : sum(depths[: (self.num_layers - 1) + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                c_multiplier=c_multiplier,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

            #############################################################

        self.num_features = int(embed_dim * c_multiplier ** (self.num_layers - 1))

        self.norm = norm_layer(self.num_features)
        # self.downdim = nn.Linear(self.num_features, 256)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.num_features = 256
        self.hidden = nn.Linear(self.num_features, 4 * self.num_features)
        self.head = nn.Linear(4 * self.num_features, num_classes)

        # encoder_hidden_size = 2048
        # input_node_num = self.last_window_size[0] * self.last_window_size[1] * self.last_window_size[2]
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.num_features *
        #               input_node_num, encoder_hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(encoder_hidden_size, encoder_hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(encoder_hidden_size,
        #               self.num_features * input_node_num),
        # )
        # self.dec = DEC(cluster_number=100, hidden_dimension=self.num_features, encoder=self.encoder,
        #                orthogonal=True, freeze_center=True, project_assignment=True)


    def forward(self, x, mask=None):

        #print model parameters
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.hape)

        if self.to_float:
            # converting tensor to float
            x = x.float()
        x = self.pos_drop(x)  # (b, c, h, w, d)

        for i in range(self.num_layers):
            x = self.pos_embeds[i](x)
            x, mask = self.layers[i](x.contiguous(), mask=mask)


        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        x = self.norm(x)  # B L C

        # x = self.downdim(x)
        #
        # x, assignment = self.dec(x)

        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = self.head(x)

        return x