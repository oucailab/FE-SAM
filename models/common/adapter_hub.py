import torch
import torch.nn as nn
from einops import rearrange
from torch.jit import script
from torch.nn.parameter import Parameter

from .layers import *


class CAA_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_attention = ChannelAttention(self.hidden_features, self.hidden_features // 4, self.hidden_features)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        xs = self.down_proj(x)
        xs = self.act(xs)

        xs = rearrange(xs, 'b h w c -> b c h w')
        b, c, h, w = xs.size()
        y = self.avg_pool(xs).view(b, c)
        y = self.c_attention(y).view(b, c, 1, 1)
        xs = xs * y.expand_as(xs)
        xs = rearrange(xs, 'b c h w -> b h w c')

        xs = self.up_proj(xs)
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


class SAA_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.s_attention = Conv2d(2, 1, kernel_size=7, bias=False)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        xs = self.down_proj(x)
        xs = self.act(xs)

        xs = rearrange(xs, 'b h w c -> b c h w')
        avg_out = torch.mean(xs, dim=1, keepdim=True)
        max_out, _ = torch.max(xs, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        xs = xs * self.sigmoid(self.s_attention(y))
        xs = rearrange(xs, 'b c h w -> b h w c')

        xs = self.up_proj(xs)
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
        self.s_attention.save_parameters(adapter_tensors, f"{prefix}.s_attention")

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]
        self.s_attention.load_parameters(state_dict, f"{prefix}.s_attention")


class SC_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.s_attention = Conv2d(2, 1, kernel_size=7, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_attention = ChannelAttention(self.hidden_features, self.hidden_features // 4, self.hidden_features)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        xs = self.down_proj(x)
        xs = self.act(xs)

        xs = rearrange(xs, 'b h w c -> b c h w')
        avg_out = torch.mean(xs, dim=1, keepdim=True)
        max_out, _ = torch.max(xs, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        xs = xs + xs * self.sigmoid(self.s_attention(y))


        b, c, h, w = xs.size()
        y = self.avg_pool(xs).view(b, c)
        y = self.c_attention(y).view(b, c, 1, 1)
        xs = xs + xs * y.expand_as(xs)
        xs = rearrange(xs, 'b c h w -> b h w c')

        xs = self.up_proj(xs)
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
        self.s_attention.save_parameters(adapter_tensors, f"{prefix}.s_attention")
        self.c_attention.save_parameters(adapter_tensors, f"{prefix}.c_attention")
        

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]
        self.s_attention.load_parameters(state_dict, f"{prefix}.s_attention")
        self.c_attention.load_parameters(state_dict, f"{prefix}.c_attention")


class CS_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.s_attention = Conv2d(2, 1, kernel_size=7, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_attention = ChannelAttention(self.hidden_features, self.hidden_features // 4, self.hidden_features)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        xs = self.down_proj(x)
        xs = self.act(xs)

        xs = rearrange(xs, 'b h w c -> b c h w')
        b, c, h, w = xs.size()
        y = self.avg_pool(xs).view(b, c)
        y = self.c_attention(y).view(b, c, 1, 1)
        xs = xs + xs * y.expand_as(xs)

        avg_out = torch.mean(xs, dim=1, keepdim=True)
        max_out, _ = torch.max(xs, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        xs = xs + xs * self.sigmoid(self.s_attention(y))
        xs = rearrange(xs, 'b c h w -> b h w c')

        xs = self.up_proj(xs)
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
        self.s_attention.save_parameters(adapter_tensors, f"{prefix}.s_attention")
        self.c_attention.save_parameters(adapter_tensors, f"{prefix}.c_attention")
        

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]
        self.s_attention.load_parameters(state_dict, f"{prefix}.s_attention")
        self.c_attention.load_parameters(state_dict, f"{prefix}.c_attention")


class BCS_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.s_attention = Conv2d(2, 1, kernel_size=7, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c_attention = ChannelAttention(self.hidden_features, self.hidden_features // 4, self.hidden_features)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        xs = self.down_proj(x)
        xs = self.act(xs)

        xs = rearrange(xs, 'b h w c -> b c h w')
        b, c, h, w = xs.size()
        y = self.avg_pool(xs).view(b, c)
        y = self.c_attention(y).view(b, c, 1, 1)
        xs_c = xs * y.expand_as(xs)

        avg_out = torch.mean(xs, dim=1, keepdim=True)
        max_out, _ = torch.max(xs, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        xs_s = xs * self.sigmoid(self.s_attention(y))

        xs = xs + xs_c + xs_s
        xs = rearrange(xs, 'b c h w -> b h w c')

        xs = self.up_proj(xs)
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
        self.s_attention.save_parameters(adapter_tensors, f"{prefix}.s_attention")
        self.c_attention.save_parameters(adapter_tensors, f"{prefix}.c_attention")
        

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]
        self.s_attention.load_parameters(state_dict, f"{prefix}.s_attention")
        self.c_attention.load_parameters(state_dict, f"{prefix}.c_attention")


class MC_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.conv1x1 = Conv2dNormReLU(3, 3, kernel_size = 1)
        self.conv3_1x1 = Conv2d(3, int(3 / mlp_ratio), kernel_size = 1)
        self.conv3x3 = Conv2dNormReLU(int(3 / mlp_ratio), 3, kernel_size = 3)
        self.conv5_1x1 = Conv2d(3, int(3 / mlp_ratio), kernel_size = 1)
        self.conv5x5 = Conv2dNormReLU(int(3 / mlp_ratio), 3, kernel_size = 5)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        xs = self.down_proj(x)
        xs = self.act(xs)

        xs = rearrange(xs, 'b h w (dh dw c) -> b c (h dh) (w dw)', dh=dw, dw=dw, c=3)
        xs_1x1 = self.conv1x1(xs)
        xs_3x3 = self.conv3x3(self.conv3_1x1(xs))
        xs_5x5 = self.conv5x5(self.conv5_1x1(xs))
        xs = xs + xs_1x1 + xs_3x3 + xs_5x5
        xs = rearrange(xs, 'b c (h dh) (w dw) -> b h w (dh dw c)', dh=dw, dw=dw, c=3)

        xs = self.up_proj(xs)
        return x + xs if self.skip_connect else xs

    def save_parameters(self, adapter_tensors, prefix):
        adapter_tensors[f"{prefix}.down_proj.weight"] = self.down_proj.weight
        adapter_tensors[f"{prefix}.up_proj.weight"] = self.up_proj.weight
        adapter_tensors[f"{prefix}.down_proj.bias"] = self.down_proj.bias
        adapter_tensors[f"{prefix}.up_proj.bias"] = self.up_proj.bias
        self.conv1x1.save_parameters(adapter_tensors, f"{prefix}.conv1x1")
        self.conv3_1x1.save_parameters(adapter_tensors, f"{prefix}.conv3_1x1")
        self.conv3x3.save_parameters(adapter_tensors, f"{prefix}.conv3x3")
        self.conv5_1x1.save_parameters(adapter_tensors, f"{prefix}.conv5_1x1")
        self.conv5x5.save_parameters(adapter_tensors, f"{prefix}.conv5x5")

    def load_parameters(self, state_dict, prefix):
        self.down_proj.weight = Parameter(state_dict[f"{prefix}.down_proj.weight"])
        self.up_proj.weight = Parameter(state_dict[f"{prefix}.up_proj.weight"])
        self.down_proj.bias = Parameter(state_dict[f"{prefix}.down_proj.bias"])
        self.up_proj.bias = Parameter(state_dict[f"{prefix}.up_proj.bias"])
        self.conv1x1.load_parameters(state_dict, f"{prefix}.conv1x1")
        self.conv3_1x1.load_parameters(state_dict, f"{prefix}.conv3_1x1")
        self.conv3x3.load_parameters(state_dict, f"{prefix}.conv3x3")
        self.conv5_1x1.load_parameters(state_dict, f"{prefix}.conv5_1x1")
        self.conv5x5.load_parameters(state_dict, f"{prefix}.conv5x5")


class MC_Adapter_SeparableConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.conv1x1 = Conv2dNormReLU(self.channel_size, self.channel_size, kernel_size=1)
        self.conv3x3 = SeparableConv(self.channel_size, self.conv_hidden_size, kernel_size=3)
        self.conv3_1x1 = Conv2dNormReLU(self.conv_hidden_size, self.channel_size, kernel_size=1)
        self.conv5x5 = SeparableConv(self.channel_size, self.conv_hidden_size, kernel_size=5)
        self.conv5_1x1 = Conv2dNormReLU(self.conv_hidden_size, self.channel_size, kernel_size=1)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        xs = self.down_proj(x)
        xs = self.act(xs)

        xs = rearrange(xs, 'b h w (dh dw c) -> b c (h dh) (w dw)', dh=dw, dw=dw, c=3)
        xs_1x1 = self.conv1x1(xs)
        xs_3x3 = self.conv3_1x1(self.conv3x3(xs))
        xs_5x5 = self.conv5_1x1(self.conv5x5(xs))
        xs = xs + xs_1x1 + xs_3x3 + xs_5x5
        xs = rearrange(xs, 'b c (h dh) (w dw) -> b h w (dh dw c)', dh=dw, dw=dw, c=3)

        xs = self.up_proj(xs)
        return x + xs if self.skip_connect else xs

    def save_parameters(self, adapter_tensors, prefix):
        adapter_tensors[f"{prefix}.down_proj.weight"] = self.down_proj.weight
        adapter_tensors[f"{prefix}.up_proj.weight"] = self.up_proj.weight
        adapter_tensors[f"{prefix}.down_proj.bias"] = self.down_proj.bias
        adapter_tensors[f"{prefix}.up_proj.bias"] = self.up_proj.bias
        self.conv1x1.save_parameters(adapter_tensors, f"{prefix}.conv1x1")
        self.conv3_1x1.save_parameters(adapter_tensors, f"{prefix}.conv3_1x1")
        self.conv3x3.save_parameters(adapter_tensors, f"{prefix}.conv3x3")
        self.conv5_1x1.save_parameters(adapter_tensors, f"{prefix}.conv5_1x1")
        self.conv5x5.save_parameters(adapter_tensors, f"{prefix}.conv5x5")

    def load_parameters(self, state_dict, prefix):
        self.down_proj.weight = Parameter(state_dict[f"{prefix}.down_proj.weight"])
        self.up_proj.weight = Parameter(state_dict[f"{prefix}.up_proj.weight"])
        self.down_proj.bias = Parameter(state_dict[f"{prefix}.down_proj.bias"])
        self.up_proj.bias = Parameter(state_dict[f"{prefix}.up_proj.bias"])
        self.conv1x1.load_parameters(state_dict, f"{prefix}.conv1x1")
        self.conv3_1x1.load_parameters(state_dict, f"{prefix}.conv3_1x1")
        self.conv3x3.load_parameters(state_dict, f"{prefix}.conv3x3")
        self.conv5_1x1.load_parameters(state_dict, f"{prefix}.conv5_1x1")
        self.conv5x5.load_parameters(state_dict, f"{prefix}.conv5x5")


class FFT_Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        channel_size: int = 3,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.conv_hidden_size = int(self.channel_size / mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()
        self.s_attention = Conv2d(2, 1, kernel_size=7, bias=False)
        self._init_weights()

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
        dw = int((self.hidden_features // 3) ** 0.5)
        assert (
            self.hidden_features == dw * dw * 3
        ), "dim mismatch for 3-channel square image"
        print(x.shape)
        xs = torch.fft.fft2(x, dim=(-3, -2))
        xs = torch.abs(xs)
        xs = self.down_proj(xs)
        xs = self.act(xs)
        xs = rearrange(xs, 'b h w c -> b c h w')

        avg_out = torch.mean(xs, dim=1, keepdim=True)
        max_out, _ = torch.max(xs, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)

        attention_weights = self.sigmoid(self.s_attention(y))
        xs = xs * attention_weights
        
        
        xs = rearrange(xs, 'b c h w -> b h w c')
        xs = self.up_proj(xs)
        xs = torch.fft.ifft2(xs, dim=(-3, -2)).real
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
        self.s_attention.save_parameters(adapter_tensors, f"{prefix}.s_attention")

    def load_parameters(self, state_dict, prefix):
        for name, module in self.named_children():
            if hasattr(module, "weight"):
                module.weight = Parameter(state_dict[f"{prefix}.{name}.weight"])
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = Parameter(state_dict[f"{prefix}.{name}.bias"])
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.data = state_dict[f"{prefix}.{name}.running_mean"]
                module.running_var.data = state_dict[f"{prefix}.{name}.running_var"]
        self.s_attention.load_parameters(state_dict, f"{prefix}.s_attention")


class New_FFT_Adapter(nn.Module):
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
        super().__init__()
        self.dims_h, self.dims_w = dims_h, dims_w
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.act = act_layer()
        self.norm = nn.LayerNorm(self.hidden_features)
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self.complex_weight = nn.Parameter(torch.randn(dims_h, dims_w, self.hidden_features, 2, dtype=torch.float32)) 
        self._init_weights()

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
        xs = self.down_proj(x)
        xs = self.act(xs)
    
        b, h, w, dim = xs.shape
        if self.dims_h == h and self.dims_w == w:
            xs = torch.fft.fft2(xs, dim=(-3, -2), norm='ortho')
            weight = torch.view_as_complex(self.complex_weight)
            xs = xs * xs * weight.unsqueeze(0)
            xs = torch.fft.ifft2(xs, s=(h, w), dim=(-3, -2), norm='ortho')
            xs = xs.reshape(b, h, w, dim)
            xs = self.norm(torch.abs(xs))
            
        xs = self.up_proj(xs)
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


class LR_FFT_Adapter(nn.Module):
    """
    Low-Rank Fast Fourier Transform Adapter (LR-FFT Adapter)
    
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
        super(LR_FFT_Adapter, self).__init__()
        self.dims_h, self.dims_w = dims_h, dims_w
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.act = act_layer()
        self.norm = nn.LayerNorm(self.hidden_features)
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        
        # Low-rank approximation of complex weights
        self.complex_weight_h = torch.nn.Parameter(torch.randn(dims_h, 1, self.hidden_features, 2, dtype=torch.float32))
        self.complex_weight_w = torch.nn.Parameter(torch.randn(1, dims_w, self.hidden_features, 2, dtype=torch.float32))
        
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
        if self.dims_h == h and self.dims_w == w:
            xs = torch.fft.fft2(xs, dim=(-3, -2), norm='ortho')
            
            # Generate full complex weight matrix using outer product
            weight = torch.view_as_complex(self.complex_weight_h * self.complex_weight_w)
            xs = xs * xs * weight
            
            xs = torch.fft.ifft2(xs, s=(h, w), dim=(-3, -2), norm='ortho')
            xs = xs.reshape(b, h, w, dim)
            xs = self.norm(torch.abs(xs))
            
        xs = self.up_proj(xs)
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


class TLR_FFT_Adapter(nn.Module):
    """
    Low-Rank Fast Fourier Transform Adapter (LR-FFT Adapter)
    
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
        super(TLR_FFT_Adapter, self).__init__()
        self.dims_h, self.dims_w = dims_h, dims_w
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.channel_size = channel_size
        self.act = act_layer()
        self.norm = nn.LayerNorm(self.hidden_features)
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        
        # Low-rank approximation of complex weights
        self.complex_weight_h = torch.nn.Parameter(torch.randn(dims_h, 1, self.hidden_features, 2, dtype=torch.float32))
        self.complex_weight_w1 = torch.nn.Parameter(torch.randn(1, dims_w, self.hidden_features, 2, dtype=torch.float32))
        self.complex_weight_w2 = torch.nn.Parameter(torch.randn(1, dims_w, self.hidden_features, 2, dtype=torch.float32))
        
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
        
        if self.dims_h == h and self.dims_w == w:
            xs_fft = torch.fft.fft2(xs, dim=(-3, -2), norm='ortho')
            
            low_freq_mask, high_freq_mask = self.adaptive_threshold(xs_fft)
            low_freq = xs_fft * low_freq_mask
            high_freq = xs_fft * high_freq_mask

            low_rank_low = torch.view_as_complex(self.complex_weight_h * self.complex_weight_w1)
            low_rank_high = torch.view_as_complex(self.complex_weight_h * self.complex_weight_w2)

            low_freq = low_freq * low_rank_low
            high_freq = high_freq * low_rank_high
            xs_fft = low_freq + high_freq

            xs = torch.fft.ifft2(xs_fft, s=(h, w), dim=(-3, -2), norm='ortho')
            xs = xs.reshape(b, h, w, dim)
            xs = self.norm(torch.abs(xs))
            
        xs = self.up_proj(xs)
        return x + xs if self.skip_connect else xs
    
    def adaptive_threshold(self, xs_fft, energy_ratio=0.1):
        b, h, w, c = xs_fft.shape
        magnitude = torch.abs(xs_fft)
    
        total_energy = magnitude.sum(dim=(-3, -2), keepdim=True)
        cumulative_energy = magnitude.cumsum(dim=-3).cumsum(dim=-2)
    
        low_freq_mask = cumulative_energy <= energy_ratio * total_energy
        high_freq_mask = ~low_freq_mask
    
        return low_freq_mask, high_freq_mask

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