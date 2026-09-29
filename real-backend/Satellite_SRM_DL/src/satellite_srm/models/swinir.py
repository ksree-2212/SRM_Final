"""Core Swin Transformer image restoration blocks and attention mechanisms."""
import math
import numpy as np
from satellite_srm.compat import torch, nn, F

class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA)."""
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        if hasattr(x, "dim") and callable(getattr(x, "dim")):
            # Pure PyTorch tensor path — fully differentiable
            is_4d = x.dim() == 4
            if is_4d:
                B, H, W, C = x.shape
                N = H * W
                x_flat = x.view(B, N, C)
            else:
                B, N, C = x.shape
                x_flat = x

            qkv = self.qkv(x_flat)
            head_dim = C // self.num_heads
            qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)

            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            result = self.proj(out)

            if is_4d:
                result = result.view(B, H, W, C)
            return result
        else:
            # NumPy fallback
            arr = x
            is_4d = arr.ndim == 4
            if is_4d:
                B, H, W, C = arr.shape
                N = H * W
            else:
                B, N, C = arr.shape

            qkv = self.qkv(torch.from_numpy(arr.reshape(B, N, C).astype(np.float32))).detach().numpy()
            head_dim = C // self.num_heads
            qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
            attn_max = np.max(attn, axis=-1, keepdims=True)
            attn_exp = np.exp(attn - attn_max)
            attn = attn_exp / (np.sum(attn_exp, axis=-1, keepdims=True) + 1e-9)

            out = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, C)
            result = self.proj(torch.from_numpy(out.astype(np.float32))).detach().numpy()

            if is_4d:
                result = result.reshape(B, H, W, C)
            return result

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with LayerNorm, MSA, and MLP."""
    def __init__(self, dim: int, num_heads: int = 6, window_size: int = 8, shift_size: int = 0, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        if hasattr(x, "dim") and callable(getattr(x, "dim")):
            # Pure PyTorch tensor path — fully differentiable
            B, H, W, C = x.shape
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size

            if pad_r > 0 or pad_b > 0:
                x_pad = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            else:
                x_pad = x

            H_pad, W_pad = x_pad.shape[1], x_pad.shape[2]
            shortcut = x_pad
            x_pad = self.norm1(x_pad)

            if self.shift_size > 0:
                shifted_x = torch.roll(x_pad, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x_pad

            # Window partition
            x_windows = shifted_x.view(B, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size, C)
            x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)

            # W-MSA
            attn_windows = self.attn(x_windows)

            # Window reverse
            attn_windows = attn_windows.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size, self.window_size, C)
            shifted_x = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, C)

            if self.shift_size > 0:
                x_pad = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x_pad = shifted_x

            x_pad = shortcut + x_pad
            x_pad = x_pad + self.mlp(self.norm2(x_pad))

            if pad_r > 0 or pad_b > 0:
                x_pad = x_pad[:, :H, :W, :].contiguous()

            return x_pad
        else:
            H, W = x.shape[1], x.shape[2]
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size

            x_pad = x
            if pad_r > 0 or pad_b > 0:
                if hasattr(x, 'numpy'):
                    x_pad = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
                else:
                    x_pad = np.pad(x, ((0,0), (0, pad_b), (0, pad_r), (0,0)), mode='constant')

            shortcut = x_pad
            x_pad = self.norm1(x_pad)

            if hasattr(x_pad, 'detach'):
                x_pad_arr = x_pad.detach().numpy()
            elif hasattr(x_pad, 'numpy'):
                x_pad_arr = x_pad.numpy()
            else:
                x_pad_arr = x_pad

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = np.roll(x_pad_arr, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x_pad_arr

        # Window partition
        B, H_pad, W_pad, C = shifted_x.shape
        x_windows = shifted_x.reshape(B, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size, C)
        x_windows_tensor = torch.from_numpy(x_windows.astype(np.float32))

        # W-MSA
        attn_windows = self.attn(x_windows_tensor)
        
        if hasattr(attn_windows, 'detach'):
            attn_windows = attn_windows.detach().numpy()
        elif hasattr(attn_windows, 'numpy'):
            attn_windows = attn_windows.numpy()

        # Window reverse
        attn_windows = attn_windows.reshape(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size, self.window_size, C)
        shifted_x = attn_windows.transpose(0, 1, 3, 2, 4, 5).reshape(B, H_pad, W_pad, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_pad = np.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x_pad = shifted_x

        x_pad = torch.from_numpy(x_pad.astype(np.float32))
        x_pad = shortcut + x_pad
        x_pad = x_pad + self.mlp(self.norm2(x_pad))

        # Unpad
        if pad_r > 0 or pad_b > 0:
            x_pad = x_pad[:, :H, :W, :].contiguous() if hasattr(x_pad, 'numpy') else x_pad[:, :H, :W, :]
            
        return x_pad

class ResidualSwinTransformerBlock(nn.Module):
    """Residual Swin Transformer Block (RSTB) encapsulating multiple Swin blocks and residual conv."""
    def __init__(self, dim: int, depth: int = 4, num_heads: int = 6, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        tokens = x.permute(0, 2, 3, 1)
        for block in self.blocks:
            tokens = block(tokens)
        out = tokens.permute(0, 3, 1, 2)
        out = self.conv(out)
        return out + res

class SwinIR(nn.Module):
    """Standard SwinIR backbone for image super-resolution."""
    def __init__(self, img_size: int = 128, in_channels: int = 3, out_channels: int = 3,
                 embed_dim: int = 96, depths=(4, 4, 4, 4), num_heads=(6, 6, 6, 6),
                 window_size: int = 8, scale: int = 3):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.rstb_layers = nn.ModuleList([
            ResidualSwinTransformerBlock(dim=embed_dim, depth=d, num_heads=h, window_size=window_size)
            for d, h in zip(depths, num_heads)
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (scale * scale), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = self.conv_first(x)
        body = feat
        for rstb in self.rstb_layers:
            body = rstb(body)
        body = self.conv_after_body(body) + feat
        out = self.upsample(body)
        return out
