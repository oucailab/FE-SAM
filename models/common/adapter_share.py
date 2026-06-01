import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.parameter import Parameter

from .layers import *
from .adapter import *
from .adapter_hub import *


# GDFN in Restormer: [github] https://github.com/swz30/Restormer
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_features: int,
        act_layer: nn.Module = nn.GELU,
        bias: bool = False
    ):
        super(FeedForward, self).__init__()

        self.act = act_layer()
        self.project_in_1 = Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.project_in_3 = Conv2d(dim, hidden_features*2, kernel_size=3, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1 = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
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

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.norm1 = nn.LayerNorm(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_attention = ChannelAttention(dim, dim * 4, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4, act_layer)
        self._init_weights()

    def _init_weights(self):
        # Initializes weights for the convolutional layers
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
        
        y = self.avg_pool(torch.abs(xs)).view(b, c)
        y = self.c_attention(y).view(b, c, 1, 1)
        xs = xs * y.expand_as(xs)
    
        xs = rearrange(xs, 'b c h w -> b h w c')

        x = x + xs if self.skip_connect else xs
        xs = self.norm2(x)
        xs = rearrange(xs, 'b h w c -> b c h w')
        xs = self.ffn(xs)
        xs = rearrange(xs, 'b c h w -> b h w c')

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
        self.c_attention.save_parameters(adapter_tensors, f"{prefix}.c_attention")
        self.ffn.save_parameters(adapter_tensors, f"{prefix}.ffn")

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]
        self.c_attention.load_parameters(state_dict, f"{prefix}.c_attention")
        self.ffn.load_parameters(state_dict, f"{prefix}.ffn")


class SWA_Adapter(nn.Module):
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
            block = Block(self.hidden_features, act_layer, skip_connect)
            self.blocks.append(block)

    def _init_weights(self):
        # Initializes weights for the linear layers
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # Initializes weights for the convolutional layers
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


class SWA_Base_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 12,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True,
    ):
        super().__init__()
        self.act_layer = act_layer
        self.skip_connect = skip_connect

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = Adapter(in_features, out_features, mlp_ratio, act_layer, skip_connect)
            self.blocks.append(block)

    def _init_weights(self):
        # Initializes weights for the linear layers
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # Initializes weights for the convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
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


class SWA_CAA_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        depth: int = 12,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True,
    ):
        super().__init__()
        self.act_layer = act_layer
        self.skip_connect = skip_connect

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = CAA_Adapter(in_features, out_features, mlp_ratio=mlp_ratio, act_layer=act_layer, skip_connect=skip_connect)
            self.blocks.append(block)

    def _init_weights(self):
        # Initializes weights for the linear layers
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        # Initializes weights for the convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
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


class Tiny_SWA_Adapter(nn.Module):
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
        self.blocks = nn.ModuleList([
            self.create_block(self.hidden_features, act_layer, skip_connect)
            for _ in range(depth)
        ])
        self.up_proj = nn.Linear(self.hidden_features, out_features)

        self._init_weights()

    def create_block(self, dim, act_layer, skip_connect):
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            act_layer(),
            nn.Linear(dim * 4, dim),
        )

    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
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
            if isinstance(module, torch.nn.LayerNorm):
                adapter_tensors[f"{prefix}.{name}.weight"] = module.weight
                adapter_tensors[f"{prefix}.{name}.bias"] = module.bias
        for i, block in enumerate(self.blocks):
            for name, module in block.named_children():
                if hasattr(module, "weight"):
                    adapter_tensors[f"{prefix}.blocks.{i}.{name}.weight"] = module.weight
                if hasattr(module, "bias") and module.bias is not None:
                    adapter_tensors[f"{prefix}.blocks.{i}.{name}.bias"] = module.bias
                if isinstance(module, torch.nn.LayerNorm):
                    adapter_tensors[f"{prefix}.blocks.{i}.{name}.weight"] = module.weight
                    adapter_tensors[f"{prefix}.blocks.{i}.{name}.bias"] = module.bias
    
    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.LayerNorm):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
        for i, block in enumerate(self.blocks):
            for name, module in block.named_children():
                if hasattr(module, "weight"):
                    module.weight = Parameter(state_dict[f"{prefix}.blocks.{i}.{name}.weight"])
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias = Parameter(state_dict[f"{prefix}.blocks.{i}.{name}.bias"])
                if isinstance(module, torch.nn.LayerNorm):
                    module.weight = Parameter(state_dict[f"{prefix}.blocks.{i}.{name}.weight"])
                    module.bias = Parameter(state_dict[f"{prefix}.blocks.{i}.{name}.bias"])