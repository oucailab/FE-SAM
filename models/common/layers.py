import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MaxPooling(nn.MaxPool2d):
    def __init__(self, kernel_size=3, dilation=1, stride=1):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super(MaxPooling, self).__init__(kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding)


class AvgPooling(nn.AvgPool2d):
    def __init__(self, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(AvgPooling, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding)


class Conv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        super(Conv2d, self).__init__(conv)

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


class Conv2dNorm(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(Conv2dNorm, self).__init__(in_channels, out_channels, kernel_size, dilation, stride, bias)
        self.add_module('norm', norm_layer(out_channels))


class Conv2dNormReLU(Conv2dNorm):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(Conv2dNormReLU, self).__init__(in_channels, out_channels, kernel_size, dilation, stride, norm_layer, bias)
        self.add_module('relu', nn.ReLU())


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, groups=in_channels, bias=False)
        pw_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        super(SeparableConv, self).__init__(dw_conv, pw_conv)

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
                module.running_mean = Parameter(state_dict[f"{prefix}.{name}.running_mean"])
                module.running_var = Parameter(state_dict[f"{prefix}.{name}.running_var"])


class SeparableConvNorm(SeparableConv):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvNorm, self).__init__(in_channels, out_channels, kernel_size, stride, dilation)
        self.add_module('norm', norm_layer(out_channels))


class SeparableConvNormReLU(SeparableConvNorm):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvNormReLU, self).__init__(in_channels, out_channels, kernel_size, stride, dilation, norm_layer)
        self.add_module('relu', nn.ReLU())


class ChannelAttention(nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ChannelAttention, self).__init__()
        self.add_module('fc1', nn.Linear(input_dim, hidden_dim, bias=False))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('fc2', nn.Linear(hidden_dim, output_dim, bias=False))
        self.add_module('sigmoid', nn.Sigmoid())

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
                module.running_mean = Parameter(state_dict[f"{prefix}.{name}.running_mean"])
                module.running_var = Parameter(state_dict[f"{prefix}.{name}.running_var"])


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x