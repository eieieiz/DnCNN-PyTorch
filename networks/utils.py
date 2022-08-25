import torch, math, copy
import opt_einsum as oe
from torch import nn
from typing import Optional
from dataclasses import dataclass
from functools import partial 
from torch.nn.modules.utils import _pair


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """
    Implementation of a squeeze and excitation layer.
    """

    def __init__(self, in_channels, ratio=16, activation=None):
        """
        - in_channels (int): Number of channels of the input.
        - ratio (int): The number of times we increase the number of channels of the intermediate representation of the input. The smaller the better but the higher associated the computation cost. Default being 16 as suggested in the original paper.
        - activation (nn.Module): A provided activation function. Default being Swish.
        """

        super().__init__()

        mid_channels = math.ceil(in_channels / ratio)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(
            in_channels, 
            mid_channels, 
            1, 
            stride=1, 
            padding=0
        )

        self.activation = Swish() if activation is None else activation()

        self.conv2 = nn.Conv2d(
            mid_channels, 
            in_channels, 
            1, 
            stride=1, 
            padding=0
        )

    def forward(self, x):
        A = self.global_pool(x)
        A = self.activation(self.conv1(A))
        A = self.conv2(A)
        A = torch.sigmoid(A)

        return x * A


class MBConv(nn.Module):
    """
    Implementation of a MBConv block, as used in EfficientNet.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        expend_ratio=2, 
        kernel_size=(3,3), 
        stride=1,
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode="zeros",
        residual=True, 
        use_se=False,
        se_ratio=16,
        activation=None,
        normalization=None
    ):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. Note that is in_channels != out_channels, no residual connection can be applied.
        - expend_ratio (int): The number of intermediary channels is computed using expend_ratio * in_channels. Default being 2 as in EfficientNet.
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, groups, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - residual (bool): Adds a residual connection from input to output.
        - use_se (bool): Toggles the usage of Squeeze and Exception. 
        - se_ration (int): The ratio of the Squeeze and Excitation layer. See the documentation of SEBlock for more details.
        - activation (nn.Module): A provided activation function. Default being Swish().
        - normalization (nn.Module): The layer to be used for normalization. Default is None, corresponding to nn.BatchNorm2d. 
        """

        super().__init__()

        mid_channels = math.ceil(in_channels * expend_ratio)
        self.use_se = use_se
        self.residual = residual and (in_channels == out_channels) and (stride == 1)

        self.normalization = nn.BatchNorm2d if normalization is None else normalization
        self.activation = nn.SiLU if activation is None else activation

        self.conv_channelwise = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False),
            self.normalization(mid_channels),
            self.activation()
        )

        self.conv_spatialwise = nn.Sequential(
            nn.Conv2d(
                mid_channels, mid_channels, kernel_size, 
                stride=stride, padding=padding, dilation=dilation,
                groups=mid_channels, bias=bias, 
                padding_mode=padding_mode
            ),
            self.normalization(mid_channels),
            self.activation()
        )

        self.se = SEBlock(
            mid_channels, 
            ratio=se_ratio, 
            activation=activation
        ) if self.use_se else nn.Identity()

        self.conv_channelwise2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            self.normalization(out_channels)
        )

    def forward(self, x):
        y = self.conv_channelwise(x)
        y = self.conv_spatialwise(y)
        y = self.se(y)
        y = self.conv_channelwise2(y)

        if self.residual:
            y = y + x

        return y


class EcoConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='zeros', 
        skip=False, groups=1, n_blocks=1):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.skip = skip

        if groups == "full":
            groups = in_channels

        if groups % in_channels != 0 or groups % out_channels != 0:
            groups = in_channels

        if self.kernel_size[0] != (1,1):
            self.layers = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(
                    in_channels, in_channels, kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                    groups=groups, bias=False, padding_mode=padding_mode
                    ),
                    nn.Conv2d(
                        in_channels, 
                        out_channels if i == n_blocks-1 else in_channels, 
                        1,
                        stride=1, padding=0, dilation=1, groups=1,
                        bias=bias, padding_mode=padding_mode
                    )
                )
                for i in range(n_blocks)]
            )
        else:
            self.layers = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias, padding_mode=padding_mode 
            )

    def forward(self, x):
        y = self.layers(x)

        if self.skip and self.in_channels == self.out_channels:
            return y + x
        return y


class CBAM(nn.Module):
    def __init__(self, in_channels, activation=None, r=2, bias=True):
        """
        The CBAM module as described in https://arxiv.org/pdf/1807.06521.pdf.
        Args:
        - in_channels (int): Number of input channels
        - activation (nn.Module): Activation function (default: nn.Identity)
        - r (float): Reduction factor
        - bias (bool): Controls if the convolutions should have bias or not
        """

        super().__init__()

        mid_channels = int(in_channels / r)
        activation = nn.Identity if activation is None else activation

        # Channel attention module
        self.maxpool_channel = nn.AdaptiveMaxPool2d((1,1))
        self.avgpool_channel = nn.AdaptiveAvgPool2d((1,1))

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=bias),
            activation(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=bias)
        )

        # Spatial attention module
        self.post_filtering = nn.Conv2d(2, 1, 7, padding=3, bias=bias)

    def forward(self, x):
        # Channel attention module
        A_channel = torch.sigmoid(
            self.shared_mlp(self.avgpool_channel(x)) + \
            self.shared_mlp(self.maxpool_channel(x))
        )
        x = A_channel * x

        # Spatial attention module
        A_spatial = torch.sigmoid(
            self.post_filtering(
                torch.cat([
                    x.mean(1, keepdim=True),
                    x.max(1, keepdim=True).values
                ], 1)
            )
        )
        x = A_spatial * x

        return x


class CBAM2(nn.Module):
    def __init__(self, in_channels, activation=None, r=2, bias=True):
        """
        The CBAM module as used in Yolov4 (https://arxiv.org/abs/2004.10934)
        Args:
        - in_channels (int): The number of input channels
        - activation (nn.Module): The activation function (default: nn.Identity)
        - r (float): Reduction factor
        - bias (bool): Controls if the convolutions should have bias or not
        """

        super().__init__()

        mid_channels = int(in_channels / r)

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=bias),
            activation(),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        A = self.layers(x)

        return A * x


class CBAM3(nn.Module):
    def __init__(self, in_channels, activation=None, r=2, bias=True):
        """
        CBAM module where the max is replaced with the std.
        Args:
        - in_channels (int): The number of input channels
        - activation (nn.Module): The activation function (deulfat: nn.Identity)
        - r (float): Reduction factor
        - bias (bool): Controls if the convolutions should have bias or not
        """

        super().__init__()

        mid_channels = int(in_channels / r)
        activation = nn.Identity if activation is None else activation

        # Channel attention module
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=bias),
            activation(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=bias)
        )

        # Spatial attention module
        self.post_filtering = nn.Conv2d(2, 1, 7, padding=3, bias=bias)

    def forward(self, x):
        # Channel attention module
        A_channel = torch.sigmoid(
            self.shared_mlp(x.mean((2,3), keepdim=True)) + \
            self.shared_mlp(x.std((2,3), keepdim=True))
        )
        x = A_channel * x

        # Spatial attention module
        A_spatial = torch.sigmoid(
            self.post_filtering(
                torch.cat([
                    x.mean(1, keepdim=True),
                    x.std(1, keepdim=True)
                ], 1)
            )
        )
        x = A_spatial * x

        return x


class MultiHeadedSpatialAttention(nn.Module):
    """
    Implementation of the multi-headed spatial attention mechanism for computer vision.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        padding=0,
        stride=1,
        dilation=1, 
        heads=1, 
        bias=True, 
        padding_mode='zeros',
        residual=True
    ):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. 
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - heads (int): Number of heads. Has to be a multiplier of in_channels and out_channels.
        - bias (bool): Controls if the convolutions should have bias or not
        """

        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = int(out_channels / heads)
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.heads = heads
        self.residual = residual and (self.in_channels == self.out_channels)

        self.to_Q = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            bias=bias
        )
        self.to_K = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            bias=bias
        )
        self.to_V = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            bias=bias
        )

        self.unfolding = nn.Unfold(
            kernel_size, 
            dilation=dilation, 
            padding=padding, 
            stride=stride
        )

    def unfold(self, x):
        B, _, H, W = x.size()

        return self.unfolding(x).reshape(
            B,
            self.heads, 
            self.mid_channels, 
            self.kernel_size[0] * self.kernel_size[1], 
            H, W
        )

    def forward(self, x):
        B, C, H, W = x.size() 

        Q = self.to_Q(x)
        K = self.to_K(x)
        V = self.to_V(x)

        K = self.unfold(K)
        V = self.unfold(V)
        Q = Q.reshape(
            B,
            self.heads, 
            self.mid_channels,
            H, W
        )

        A = torch.einsum("b g c h w, b g c p h w -> b g p h w", Q, K)
        A = A.softmax(2)

        y = torch.einsum("b g p h w, b g c p h w -> b g c h w", A, V)
        y = y.reshape(B, C, H, W)

        if self.residual:
            y = x + y

        return y


class MultiHeadedGlobalAttention(nn.Module):
    """
    Implementation of the multi-headed attention mechanism for computer vision.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        padding=0,
        stride=1,
        dilation=1, 
        heads=1, 
        bias=True, 
        padding_mode='zeros',
        residual=True
    ):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. 
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - heads (int): Number of heads. Has to be a multiplier of in_channels and out_channels.
        - bias (bool): Controls if the convolutions should have bias or not
        """

        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = int(out_channels / heads)
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.heads = heads
        self.residual = residual and (self.in_channels == self.out_channels)

        self.to_Q = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            bias=bias
        )
        self.to_K = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            bias=bias
        )
        self.to_V = nn.Conv2d(
            in_channels, 
            out_channels, 
            1, 
            bias=bias
        )

    def forward(self, x):
        B, C, H, W = x.size() 

        Q = self.to_Q(x)
        K = self.to_K(x)
        V = self.to_V(x)

        Q = Q.reshape(B, self.heads, self.mid_channels, H*W)
        K = K.reshape(B, self.heads, self.mid_channels, H*W)
        V = V.reshape(B, self.heads, self.mid_channels, H*W)

        A = torch.einsum("b h c n, b h c m -> b h n m", Q, K)
        A = A.softmax(3)

        y = torch.einsum("b h n m, b h c m -> b h c n", A, V)
        y = y.reshape(B, C, H, W)

        if self.residual:
            y = x + y

        return y


class NonLocalMean(nn.Module):
    """
    Implementation of a simple non-local mean module that operates on a patch.
    """

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        padding=0,
        stride=1,
        dilation=1, 
        heads=1, 
        bias=True, 
        padding_mode='zeros',
        residual=True
    ):
        """
        - in_channels (int): Number of channels of the input.
        - out_channels (int): Number of desired channels for the output. 
        - kernel_size (int): The kernel size of the convolution layer.
        - stride, padding, dilation, bias, padding_mode: see the nn.Conv2d documentation of PyTorch.
        - heads (int): Number of heads. Has to be a multiplier of in_channels and out_channels.
        - bias (bool): Controls if the convolutions should have bias or not
        """

        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = int(out_channels / heads)
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.heads = heads
        self.residual = residual

        self.unfolding = nn.Unfold(
            kernel_size, 
            dilation=dilation, 
            padding=padding, 
            stride=stride
        )

    def unfold(self, x):
        B, _, H, W = x.size()

        return self.unfolding(x).reshape(
            B,
            self.heads, 
            self.mid_channels, 
            self.kernel_size[0] * self.kernel_size[1], 
            H, W
        )

    def forward(self, x):
        B, C, H, W = x.size() 

        x_unf = self.unfold(x)
        x_r = x.reshape(
            B,
            self.heads, 
            self.mid_channels,
            H, W
        )

        # ||K-Q||^2_2
        A = torch.pow(x_unf, 2).sum(2) \
            + torch.pow(x_r, 2).sum(2, keepdim=True) \
            - 2*torch.einsum("b g c p h w, b g c h w -> b g p h w", x_unf, x_r)

        A = torch.softmax(-A, 2)

        y = torch.einsum("b g p h w, b g c p h w -> b g c h w", A, x_unf)
        y = y.reshape(B, C, H, W)

        if self.residual:
            y = x + y

        return y


class HuaweiMaskExtractor(nn.Module):
    def __init__(self, 
        in_channels, mid_channels=64, 
        depth=4, 
        normalization=None, activation=None,
        mode="concat",
        copy=False
    ):
        """
        Args:
            - in_channels (int): Number of input channels.
            - mid_channels (int): Number of intermediate channels.
            - depth (int): Number of convolutions to be used in total.
            - normalization (nn.Module): Normalization layer. Default is nn.Identity
            - activation (nn.Module): Activation layer. Default is nn.ReLU
            - mode (string): Mode of the layer. It can be "multiply" or "concat".
            - copy (bool): Whether or not to use the "copy mask" formula. Only available in "multiply" mode.
        """

        super().__init__()

        assert mode in ["multiply", "concat"], "The specified mode doesn't exist. It should be either `multiply` or `concat`."
        assert not copy or mode == "multiply", "Copy mode is only available in multiply mode."

        activation = nn.ReLU if activation is None else activation
        normalization = nn.Identity if normalization is None else normalization
        self.mode = mode
        self.copy = copy

        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                normalization(mid_channels),
                activation()
            )
        )

        for _ in range(depth-2):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                    normalization(mid_channels),
                    activation()
                )
            )

        layers.append(
            nn.Sequential(
                nn.Conv2d(mid_channels, 1, 3, padding=1),
                nn.Sigmoid()
            )
        )

        self.submodel = nn.Sequential(*layers)

    def forward(self, x):
        mask = self.submodel(x)

        if self.mode == "multiply":
            a, b = torch.chunk(x, 2, dim=1)

            if self.copy:
                return torch.cat([(1-mask) * a + mask * b, b], 1)

            return torch.cat([a * mask, b], 1)

        elif self.mode == "concat":
            return torch.cat([x, mask], 1)


class SharedPreprocessing(nn.Module):
    def __init__(
        self, in_channels,
        mid_channels=48, depth=4,
        normalization=None, activation=None
    ):
        """
        Args:
            - in_channels (int): Number of input channels.
            - mid_channels (int): Numer of channels using internally.
            - depth (int): Number of convolutional layers to use.
            - normalization (nn.Module): Normalization layer to use.
            - activation (nn.Module): Activation layer to use.
        """
        super().__init__()

        normalization = nn.Identity if normalization is None else normalization
        activation = nn.ReLU if activation is None else activation 

        r_in_channels = int(in_channels//2)
        out_channels = int(mid_channels//2)

        layers = []

        layers.append(nn.Sequential(
            nn.Conv2d(r_in_channels, mid_channels, 3, padding=1),
            normalization(mid_channels),
            activation()
        ))

        for _ in range(depth-2):
            layers.append(nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                normalization(mid_channels),
                activation()
            ))

        layers.append(nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            normalization(out_channels),
            activation()
        ))

        self.skip_conv = nn.Conv2d(r_in_channels, out_channels, 1)

        self.submodel = nn.Sequential(*layers)

    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)

        a, b = map(lambda x: self.submodel(x) + self.skip_conv(x), [a, b])

        return torch.cat([a, b], 1)


class TemporalAttention(nn.Module):
    def __init__(
        self, in_channels, 
        sp_features=48, sp_depth=4,
        kernel_size=7, padding=3,
        heads=1,
        normalization=None, activation=None
    ):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.heads = heads

        self.sp = SharedPreprocessing(
            in_channels,
            mid_channels=sp_features,
            depth=sp_depth,
            normalization=normalization,
            activation=activation
        )

        r_mid_channels = int(sp_features//2)

        self.to_Q = nn.Conv2d(r_mid_channels, r_mid_channels, 1)
        self.to_K = nn.Conv2d(r_mid_channels, r_mid_channels, 1)
        self.to_V = nn.Conv2d(r_mid_channels, r_mid_channels, 1)

        self.unfolding = nn.Unfold(
            kernel_size,  
            padding=padding
        )

    def unfold(self, x):
        B, _, H, W = x.size()

        return self.unfolding(x).reshape(
            B,
            self.heads, 
            -1, 
            self.kernel_size[0] * self.kernel_size[1], 
            H, W
        )

    def forward(self, x):
        B, _, H, W = x.size() 

        # a: old, b: new
        z = self.sp(x)
        a, b = torch.chunk(z, 2, dim=1)

        Q = self.to_Q(b)
        K = self.to_K(a)
        V = self.to_V(a)

        K = self.unfold(K)
        V = self.unfold(V)
        Q = Q.reshape(
            B,
            self.heads, 
            -1,
            H, W
        )

        A = torch.einsum("b g c h w, b g c p h w -> b g p h w", Q, K) / math.sqrt(Q.size(2))
        A = A.softmax(2)

        y = torch.einsum("b g p h w, b g c p h w -> b g c h w", A, V)
        y = y.reshape(B, -1, H, W)

        return torch.cat([y + a, b], 1)


# Feature recurrence
def register_feature_hook(self, register_from, add_to, n_channels):
    self.saved_features[register_from] = []
    self.saved_features_default[register_from] = n_channels
    self.saved_features_map[add_to] = register_from

    register_from.register_forward_hook(self.save_feature_hook)
    add_to.register_forward_pre_hook(self.append_feature_hook)

def save_feature_hook(self, layer, input, output):
    self.temp_saved_features[layer] = output

def append_feature_hook(self, layer, input):
    src_layer = self.saved_features_map[layer]

    # Create default
    if len(self.saved_features[src_layer]) != self.n_saved_frames:
        B, _, H, W = input[0].size()
        C = self.saved_features_default[src_layer]

        default = torch.zeros(B, C, H, W).type(input[0].type()).to(input[0].device)
        self.saved_features[src_layer] = [default] * self.n_saved_frames

    return torch.cat((input[0], *self.saved_features[src_layer]), 1)

def update_saved_features(self):
    for layer, feature in self.temp_saved_features.items():
        del self.saved_features[layer][0]
        self.saved_features[layer].append(feature)

def reset_saved_features(self):
    for layer in self.saved_features.keys():
        self.saved_features[layer] = []

def saved_features_apply_(self, fn):
    saved_features_copy = copy.copy(self.saved_features)

    for layer, feature_group in self.saved_features.items():
        for idx, feature in enumerate(feature_group):
            saved_features_copy[layer][idx] = fn(feature, layer, idx)

    self.saved_features = saved_features_copy


class FeatureRecurrentNetwork(type):
    def __call__(cls, *args, **kwargs):
        cls.register_feature_hook = register_feature_hook
        cls.save_feature_hook = save_feature_hook
        cls.append_feature_hook = append_feature_hook
        cls.update_saved_features = update_saved_features
        cls.reset_saved_features = reset_saved_features
        cls.saved_features_apply_ = saved_features_apply_

        old_forward = cls.forward

        def forward_wrapper(self, x):
            output = old_forward(self, x)
            self.update_saved_features()

            return output
        cls.forward = forward_wrapper

        obj = cls.__new__(cls)

        obj.saved_features = {}
        obj.temp_saved_features = {}
        obj.saved_features_map = {}
        obj.saved_features_default = {}

        obj.__init__(*args, **kwargs)

        return obj


@dataclass(eq=False)
class LayerNorm(nn.Module):
    in_channels: int
    eps: float = 1e-6

    def __post_init__(self) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(self.in_channels))
        self.bias = nn.Parameter(torch.zeros(self.in_channels))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)

        return einsum("c, b c ... -> b c ...", self.weight, x) + pad_as(self.bias, x)


@dataclass(eq=False)
class LayerScale(nn.Module):
    in_channels: int
    layerscale_init: Optional[float] = 0.1

    def __post_init__(self) -> None:
        super().__init__()

        if self.layerscale_init is not None: 
            self.layerscale = nn.Parameter(self.layerscale_init * torch.ones(self.in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layerscale_init is not None:
            return einsum("c, b c ... -> b c ...", self.layerscale, x)

        return x

def pad_as(x, ref):
    _, _, *dims = ref.size()

    for _ in range(len(dims)):
        x = x.unsqueeze(dim=-1)

    return x

def zero_pad_features(size, x):
    tmp = torch.zeros(size).to(x.device)

    start_x = int((tmp.size(2) - x.size(2)) / 2)
    start_y = int((tmp.size(3) - x.size(3)) / 2)

    i1, i2 = start_x, start_x + x.size(2)
    j1, j2 = start_y, start_y + x.size(3)
    tmp[:, :, i1:i2, j1:j2] = x

    return tmp

einsum = partial(oe.contract, backend="torch")
