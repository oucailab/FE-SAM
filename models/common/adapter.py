import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
from .layers import *


class Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        xs = self.down_proj(x)
        xs = self.act(xs)
        xs = self.up_proj(xs)
        return x + xs if self.skip_connect else xs

    def save_parameters(self, adapter_tensors, prefix):
        adapter_tensors[f"{prefix}.down_proj.weight"] = self.down_proj.weight
        adapter_tensors[f"{prefix}.up_proj.weight"] = self.up_proj.weight
        adapter_tensors[f"{prefix}.down_proj.bias"] = self.down_proj.bias
        adapter_tensors[f"{prefix}.up_proj.bias"] = self.up_proj.bias

    def load_parameters(self, state_dict, prefix):
        self.down_proj.weight = Parameter(state_dict[f"{prefix}.down_proj.weight"])
        self.up_proj.weight = Parameter(state_dict[f"{prefix}.up_proj.weight"])
        self.down_proj.bias = Parameter(state_dict[f"{prefix}.down_proj.bias"])
        self.up_proj.bias = Parameter(state_dict[f"{prefix}.up_proj.bias"])


class FLRA_Adapter(nn.Module):
    """
    Frequency Low-Rank Adaptive Adapter (FLRA Adapter)

    This adapter uses a low-rank approximation of the complex weight matrix
    in combination with Fast Fourier Transform for efficient feature transformation.
    It significantly reduces the number of parameters while maintaining
    expressive power through the use of outer products.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dims_h: int = 64,
        dims_w: int = 64,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super(FLRA_Adapter, self).__init__()
        self.dims_h, self.dims_w = dims_h, dims_w
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.act = act_layer()
        self.norm = nn.LayerNorm(self.hidden_features)
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)

        self.complex_weight_h = torch.nn.Parameter(torch.randn(self.hidden_features, dims_h, 1, 2, dtype=torch.float32))
        self.complex_weight_w1 = torch.nn.Parameter(torch.randn(self.hidden_features, 1, dims_w, 2, dtype=torch.float32))
        self.complex_weight_w2 = torch.nn.Parameter(torch.randn(self.hidden_features, 1, dims_w, 2, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xs = self.down_proj(x)
        xs = self.act(xs)
        b, h, w, dim = xs.shape
        xs = rearrange(xs, 'b h w c -> b c h w')

        if self.dims_h == h and self.dims_w == w:
            xs_fft = torch.fft.fft2(xs, dim=(-2, -1), norm='ortho')
            fft_shifted = torch.fft.fftshift(xs_fft)
            magnitude_spectrum = torch.abs(fft_shifted)

            cutoff_freq = self.adaptive_cutoff_freq(magnitude_spectrum)
            low_pass_mask, high_pass_mask = self.adaptive_threshold(xs_fft, cutoff_freq)

            magnitude_spectrum_log = torch.log(magnitude_spectrum + 1)
            xs_fft_low_freq = magnitude_spectrum_log * low_pass_mask
            xs_fft_high_freq = magnitude_spectrum_log * high_pass_mask

            low_rank_low = torch.view_as_complex(self.complex_weight_h * self.complex_weight_w1)
            low_rank_high = torch.view_as_complex(self.complex_weight_h * self.complex_weight_w2)

            xs_fft_low_freq = xs_fft_low_freq * low_rank_low
            xs_fft_high_freq = xs_fft_high_freq * low_rank_high
            xs_fft = xs_fft_low_freq + xs_fft_high_freq

            xs = torch.fft.ifft2(xs_fft, s=(h, w), dim=(-2, -1), norm='ortho')
            xs = rearrange(xs, 'b c h w -> b h w c')
            xs = torch.abs(xs)
        else:
            xs = rearrange(xs, 'b c h w -> b h w c')

        xs = self.up_proj(xs)
        return x + xs if self.skip_connect else xs

    def adaptive_threshold(self, xs_fft, cutoff_freq=30):
        b, h, w, c = xs_fft.shape
        center = (h//2, w//2)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        dist_from_center = torch.sqrt((x - center[1])**2 + (y - center[0])**2).to(xs_fft.device)
        low_pass_mask = (dist_from_center <= cutoff_freq).float()
        high_pass_mask = 1 - low_pass_mask
        return low_pass_mask.unsqueeze(0).unsqueeze(-1), high_pass_mask.unsqueeze(0).unsqueeze(-1)

    def adaptive_cutoff_freq(self, magnitude_spectrum, energy_percentage=0.3):
        total_energy = torch.sum(magnitude_spectrum)
        energy_sum = 0
        h, w = magnitude_spectrum.shape[-2:]
        center = (h // 2, w // 2)

        for r in range(min(h // 2, w // 2)):
            y, x = torch.meshgrid(torch.arange(-r, r+1), torch.arange(-r, r+1), indexing='ij')
            mask = x*x + y*y <= r*r
            energy_sum += torch.sum(magnitude_spectrum[..., center[0]-r:center[0]+r+1,
                                                        center[1]-r:center[1]+r+1][..., mask])
            if energy_sum / total_energy >= energy_percentage:
                return r
        return min(h // 2, w // 2)

    def save_parameters(self, adapter_tensors, prefix):
        adapter_tensors[f"{prefix}.down_proj.weight"] = self.down_proj.weight
        adapter_tensors[f"{prefix}.up_proj.weight"] = self.up_proj.weight
        adapter_tensors[f"{prefix}.down_proj.bias"] = self.down_proj.bias
        adapter_tensors[f"{prefix}.up_proj.bias"] = self.up_proj.bias
        adapter_tensors[f"{prefix}.complex_weight_h"] = self.complex_weight_h
        adapter_tensors[f"{prefix}.complex_weight_w1"] = self.complex_weight_w1
        adapter_tensors[f"{prefix}.complex_weight_w2"] = self.complex_weight_w2

    def load_parameters(self, state_dict, prefix):
        self.down_proj.weight = Parameter(state_dict[f"{prefix}.down_proj.weight"])
        self.up_proj.weight = Parameter(state_dict[f"{prefix}.up_proj.weight"])
        self.down_proj.bias = Parameter(state_dict[f"{prefix}.down_proj.bias"])
        self.up_proj.bias = Parameter(state_dict[f"{prefix}.up_proj.bias"])
        self.complex_weight_h = Parameter(state_dict[f"{prefix}.complex_weight_h"])
        self.complex_weight_w1 = Parameter(state_dict[f"{prefix}.complex_weight_w1"])
        self.complex_weight_w2 = Parameter(state_dict[f"{prefix}.complex_weight_w2"])


class SW_Block(nn.Module):
    def __init__(
        self,
        dim: int,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.norm1 = nn.LayerNorm(dim)

        self.conv1x1 = nn.Conv2d(dim, dim * 2, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_avg = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_max = nn.Conv2d(dim * 2, dim * 2, kernel_size=1)

        self.fusion_conv = nn.Conv2d(dim * 4, dim, kernel_size=1)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            act_layer(),
            nn.Linear(dim // 2, dim // 2),
            act_layer(),
            nn.Linear(dim // 2, dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, h, w, c = x.size()
        xs = self.norm1(x)

        xs = rearrange(xs, 'b h w c -> b c h w')
        xs = self.conv1x1(xs)

        avg_branch = self.avg_pool(xs)
        avg_branch = self.conv_avg(avg_branch)
        avg_filter = F.softmax(avg_branch, dim=1)

        max_branch = self.max_pool(xs)
        max_branch = self.conv_max(max_branch)
        max_filter = F.softmax(max_branch, dim=1)

        avg_features = xs * avg_filter
        max_features = xs * max_filter

        xs = torch.cat([avg_features, max_features], dim=1)
        xs = self.fusion_conv(xs)
        xs = rearrange(xs, 'b c h w -> b h w c')

        x = x + xs if self.skip_connect else xs

        xs = self.norm2(x)
        xs = self.mlp(xs)

        return x + xs if self.skip_connect else xs

    def save_parameters(self, adapter_tensors, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                adapter_tensors[f"{prefix}.{name}.weight"] = module.weight
            if hasattr(module, "bias") and module.bias is not None:
                adapter_tensors[f"{prefix}.{name}.bias"] = module.bias
            if isinstance(module, torch.nn.BatchNorm2d):
                adapter_tensors[f"{prefix}.{name}.running_mean"] = module.running_mean
                adapter_tensors[f"{prefix}.{name}.running_var"] = module.running_var

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]


class SW_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 4,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True,
    ):
        super().__init__()
        self.act_layer = act_layer
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = SW_Block(self.hidden_features, act_layer, skip_connect)
            self.blocks.append(block)

    def _init_weights(self):
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.down_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.up_proj(x)
        return x

    def save_parameters(self, adapter_tensors, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                adapter_tensors[f"{prefix}.{name}.weight"] = module.weight
            if hasattr(module, "bias") and module.bias is not None:
                adapter_tensors[f"{prefix}.{name}.bias"] = module.bias
            if isinstance(module, torch.nn.BatchNorm2d):
                adapter_tensors[f"{prefix}.{name}.running_mean"] = module.running_mean
                adapter_tensors[f"{prefix}.{name}.running_var"] = module.running_var
        for i in range(len(self.blocks)):
            self.blocks[i].save_parameters(adapter_tensors, f"{prefix}.blocks.{i}")

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]
        for i in range(len(self.blocks)):
            self.blocks[i].load_parameters(state_dict, f"{prefix}.blocks.{i}")


class _adapter_attn(nn.Module):
    def __init__(
        self,
        block_attn_proj: nn.Module,
        adapter_attn: nn.Module,
    ):
        super().__init__()
        assert isinstance(block_attn_proj, nn.Module) and isinstance(adapter_attn, nn.Module), \
            "block_attn_proj and adapter_attn must be instances of nn.Module"
        self.proj = block_attn_proj
        self.adapter_attn = adapter_attn

    def forward(self, x):
        x = self.proj(x)
        x = self.adapter_attn(x)
        return x


class _adapter_mlp(nn.Module):
    def __init__(
        self,
        block_mlp: nn.Module,
        adapter_mlp: nn.Module,
        scale: float = 0.5,
    ):
        super().__init__()
        assert isinstance(block_mlp, nn.Module) and isinstance(adapter_mlp, nn.Module), \
            "block_mlp and adapter_mlp must be instances of nn.Module"
        self.adapter = adapter_mlp

        self.scale = scale
        self.lin1 = block_mlp.lin1
        self.lin2 = block_mlp.lin2
        self.act = block_mlp.act
        self.adapter_mlp = adapter_mlp

    def forward(self, x):
        ax = self.adapter(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x + ax * self.scale
