import torch
import torch.nn as nn 
import torch.nn.functional as F

from posetail.posetail.utils import init_kwargs


class ResUnit(nn.Module):

    def __init__(self, input_dim, output_dim, 
                 kernel_size = 3, spatial_res_factor = 1,
                 act_class = nn.ReLU, norm_class = nn.InstanceNorm2d,
                 act_kwargs = None, norm_kwargs = None):
        super().__init__()

        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.spatial_res_factor = spatial_res_factor

        self.act_class = act_class 
        self.norm_class = norm_class 
        self.act_kwargs = init_kwargs(act_kwargs)
        self.norm_kwargs = init_kwargs(norm_kwargs)

        self.conv1_branch1 = nn.Conv2d(
            in_channels = self.input_dim, 
            out_channels = self.output_dim, 
            kernel_size = self.kernel_size, 
            stride = self.spatial_res_factor, 
            padding = self.padding
        )

        self.norm1_branch1 = self.norm_class(num_features = self.output_dim,
                                             **self.norm_kwargs)

        self.act = self.act_class(**self.act_kwargs)

        self.conv2_branch1 = nn.Conv2d(
            in_channels = self.output_dim, 
            out_channels = self.output_dim, 
            kernel_size = self.kernel_size, 
            stride = 1, 
            padding = self.padding
        )

        self.norm2_branch1 = self.norm_class(num_features = self.output_dim, 
                                            **self.norm_kwargs)

        if self.spatial_res_factor > 1: 

            self.conv1_branch2 = nn.Conv2d(
                in_channels = self.input_dim, 
                out_channels = self.output_dim, 
                kernel_size = 1,
                stride = self.spatial_res_factor 
            )

            self.norm1_branch2 = self.norm_class(num_features = self.output_dim,
                                                 **self.norm_kwargs)


    def forward(self, x): 
        
        x_res = x
        
        x = self.act(self.norm1_branch1(self.conv1_branch1(x)))
        x = self.act(self.norm2_branch1(self.conv2_branch1(x)))

        if self.spatial_res_factor > 1: 
            x_res = self.norm1_branch2(self.conv1_branch2(x_res))

        x = self.act(x + x_res)

        return x


class ResBlock(nn.Module):

    def __init__(self, input_dim, output_dim, n_units = 2,
                 kernel_size = 3, spatial_res_factor = 1,
                 act_class = nn.ReLU, norm_class = nn.InstanceNorm2d,
                 act_kwargs = None, norm_kwargs = None):        
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.spatial_res_factor = spatial_res_factor
        self.n_units = n_units

        self.act_class = act_class
        self.norm_class = norm_class
        self.act_kwargs = init_kwargs(act_kwargs)
        self.norm_kwargs = init_kwargs(norm_kwargs)
        
        self.res_units = self._init_res_units()

    def _init_res_units(self): 

        res_units = nn.ModuleList()
        input_dim = self.input_dim

        for i in range(self.n_units): 

            if i == 0:
                stride = self.spatial_res_factor
            else: 
                stride = 1

            res_unit = ResUnit(
                input_dim = input_dim, 
                output_dim = self.output_dim, 
                kernel_size = self.kernel_size, 
                spatial_res_factor = stride,
                act_class = self.act_class, 
                norm_class = self.norm_class,
                act_kwargs = self.act_kwargs, 
                norm_kwargs = self.norm_kwargs
            )
            res_units.append(res_unit)
            input_dim = self.output_dim

        return res_units

    def forward(self, x): 
        
        for i in range(len(self.res_units)):
            x = self.res_units[i](x)

        return x


class ResidualFeatureExtractor(nn.Module):

    def __init__(self, input_dim, output_dim, n_blocks, 
                downsample_factor = 4, n_units = 2, interp_mode = 'bilinear', 
                kernel_size = 3, spatial_res_factor = 1, 
                act_class = nn.ReLU, norm_class = nn.InstanceNorm2d,
                act_kwargs = None, norm_kwargs = None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_blocks = n_blocks
        self.n_units = n_units
        self.downsample_factor = downsample_factor
        self.interp_mode = interp_mode
        self.kernel_size = kernel_size
        self.spatial_res_factor = spatial_res_factor

        self.act_class = act_class 
        self.norm_class = norm_class
        self.act_kwargs = init_kwargs(act_kwargs)
        self.norm_kwargs = init_kwargs(norm_kwargs)

        self.conv1 = nn.Conv2d(
            in_channels = self.input_dim, 
            out_channels = self.output_dim // 2, 
            kernel_size = 7, 
            stride = 2, 
            padding = 3)

        self.norm1 = self.norm_class(num_features = self.output_dim // 2,
                                    **self.norm_kwargs)
        self.act = self.act_class(**self.act_kwargs)

        # initialize residual blocks
        hidden_dims = [output_dim // 2, 
                       3 * (output_dim // 4), 
                       output_dim, 
                       output_dim]

        self.res_blocks = self._init_res_blocks(
            input_dim = self.output_dim // 2, 
            output_dims = hidden_dims
        )

        self.cat_dim = sum(hidden_dims)

        self.conv2 = nn.Conv2d(
            in_channels = self.cat_dim, 
            out_channels = self.output_dim * 2, 
            kernel_size = 3, 
            stride = 1,
            padding = 1
        )
        self.norm2 = self.norm_class(num_features = self.output_dim * 2,
                                    **self.norm_kwargs)

        self.conv3 = nn.Conv2d(
            in_channels = self.output_dim * 2, 
            out_channels = self.output_dim, 
            kernel_size = 1, 
            stride = 1,
            padding = 0
        )

        self._init_weights()

    def _init_weights(self): 

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d): 
                nn.init.kaiming_uniform_(layer.weight, 
                    mode = 'fan_out', nonlinearity = 'relu')
            
            elif isinstance(layer, nn.InstanceNorm2d): 
                if layer.weight is not None and layer.bias is not None: 
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def _init_res_blocks(self, input_dim, output_dims): 

        assert len(output_dims) == self.n_blocks
        res_blocks = nn.ModuleList()

        for i in range(self.n_blocks): 

            if i == 0:
                stride = 1
            else: 
                stride = self.spatial_res_factor

            res_block = ResBlock(
                input_dim = input_dim, 
                output_dim = output_dims[i], 
                spatial_res_factor = stride,
                act_class = self.act_class, 
                norm_class = self.norm_class,
                **self.act_kwargs, 
                **self.norm_kwargs
            )
            input_dim = output_dims[i]
            res_blocks.append(res_block)
            
        return res_blocks

    def resize(self, feature_maps, res):

        spatial_res = (res[0] // self.downsample_factor, 
                       res[1] // self.downsample_factor)

        feature_maps = F.interpolate(
            input = feature_maps, 
            size = spatial_res,
            mode = self.interp_mode, 
            align_corners = True
        )

        return feature_maps

    def forward(self, x): 

        target_res = x.shape[-2:]

        x = self.act(self.norm1(self.conv1(x)))

        x_scales = []

        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)
            x_resized = self.resize(x, res = target_res)
            x_scales.append(x_resized)

        x = torch.cat(x_scales, dim = 1)

        x = self.act(self.norm2(self.conv2(x)))
        x = self.conv3(x)

        return x



class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels = 128, 
                 stride = 1, downsample_factor = 2):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = 64,
            kernel_size = 3, 
            stride = stride, 
            padding = 1
        )

        self.pool = nn.MaxPool2d(
            kernel_size = downsample_factor, 
            stride = downsample_factor
        )

        self.conv2 = nn.Conv2d(
            in_channels = 64, 
            out_channels = out_channels,
            kernel_size = 3, 
            stride = 1, 
            padding = 1
        )

        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self): 

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d): 
                nn.init.kaiming_uniform_(layer.weight, 
                    mode = 'fan_out', nonlinearity = 'relu')
            
            elif isinstance(layer, nn.InstanceNorm2d): 
                if layer.weight is not None and layer.bias is not None: 
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)

        return x


from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict
import hiera
from einops import rearrange

class HieraFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128, pretrained_model='facebook/hiera_base_224.mae_in1k'):
        super().__init__()
        
        self.model = hiera.Hiera.from_pretrained(pretrained_model)

        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = True

        # Remove unused Hiera head components
        self.model.head = nn.Identity()
        self.model.norm = nn.Identity()
            
        with torch.no_grad():  # Don't build computation graph during init
            test_input = torch.randn([1, 3, 224, 224])
            _, intermediates = self.model(test_input, return_intermediates=True)
        hiera_channels = [x.shape[-1] for x in intermediates]

        self.fpn = FeaturePyramidNetwork(hiera_channels, output_dim)

        # Freeze unused FPN layer_blocks (keep only layer_blocks[0])
        for name, param in self.fpn.named_parameters():
            if 'layer_blocks' in name and not name.startswith('layer_blocks.0'):
                param.requires_grad = False

    def forward(self, x):
        _, intermediates = self.model(x, return_intermediates=True)
        features = OrderedDict()
        for i, x in enumerate(intermediates):
            features[i] = rearrange(x, 'b h w c -> b c h w')
        out = self.fpn(features)
        return out[0]

from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2HieraFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128, pretrained_model="facebook/sam2.1-hiera-small",
                 freeze_nonlast_fpn=True):
        super().__init__()

        predictor = SAM2ImagePredictor.from_pretrained(pretrained_model)
        self.model = predictor.model.image_encoder.trunk
        
        # Ensure all parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = False
            device = param.device
            
        # with torch.no_grad():  # Don't build computation graph during init
        #     test_input = torch.randn([1, 3, 400, 400]).to(device)
        #     intermediates = self.model(test_input)
        # hiera_channels = [x.shape[-1] for x in intermediates]

        if pretrained_model == 'facebook/sam2.1-hiera-base-plus':
            hiera_channels = [112, 224, 448, 896]
        elif pretrained_model in ['facebook/sam2.1-hiera-small', 'facebook/sam2.1-hiera-tiny']:
            hiera_channels = [96, 192, 384, 768]
        self.fpn = FeaturePyramidNetwork(hiera_channels, output_dim)

        if freeze_nonlast_fpn:
            # Freeze unused FPN layer_blocks (keep only layer_blocks[0])
            for name, param in self.fpn.named_parameters():
                if 'layer_blocks' in name and not name.startswith('layer_blocks.0'):
                    param.requires_grad = False

    def forward(self, x, return_all=False):
        intermediates = self.model(x)
        features = OrderedDict()
        for i, x in enumerate(intermediates):
            features[i] = x
        out = self.fpn(features)
        if return_all:
            return list(out.values())
        else:
            return out[0]
    
    

class TriplaneFeatureExtractor(nn.Module):

    def __init__(self, input_dim, n_hidden_layers, 
                 kernel_size = 3, padding = 1, 
                 upsample_factor = 4):
        super().__init__()

        self.upsample_factor = upsample_factor

        if self.upsample_factor > 1: 
            self.upsample = nn.Upsample(
                scale_factor = self.upsample_factor, 
                mode = 'bilinear',
                align_corners = True
            )

        # hidden layers
        self.hidden_conv_layers = nn.ModuleList()
        self.gn_layers = nn.ModuleList()

        for i in range(n_hidden_layers): 

            hidden_layer = nn.Conv2d(
                in_channels = input_dim, 
                out_channels = input_dim, 
                kernel_size = kernel_size,
                padding = padding
            )

            self.hidden_conv_layers.append(hidden_layer)

            gn = nn.GroupNorm(num_groups = 8, num_channels = input_dim)
            self.gn_layers.append(gn)

        self.relu = nn.ReLU()

        self.conv_out = nn.Conv2d(
            in_channels = input_dim, 
            out_channels = input_dim, 
            kernel_size = kernel_size, 
            padding = padding
        )

        self._init_weights()

    def _init_weights(self): 

        for layer in self.modules():

            if isinstance(layer, nn.Conv2d): 
                nn.init.kaiming_uniform_(layer.weight, 
                    mode = 'fan_out', nonlinearity = 'relu')

    def forward(self, x):

        if self.upsample_factor > 1:
            x = self.upsample(x)

        # for conv in self.hidden_conv_layers: 
        #     x = self.relu(conv(x))

        for gn, conv in zip(self.gn_layers, self.hidden_conv_layers): 
            x = self.relu(gn(conv(x)))

        x = self.conv_out(x)
        
        return x


class FeatureUpdater(nn.Module):

    def __init__(self, latent_dim):
        super().__init__()
 
        self.latent_dim = latent_dim

        self.norm = nn.GroupNorm(
            num_groups = 1,
            num_channels = self.latent_dim
        )

        self.fc1 = nn.Linear(
            in_features = latent_dim, 
            out_features = latent_dim
        )

        self.gelu = nn.GELU()

    def forward(self, x): 

        x = self.norm(x)
        x = self.fc1(x)
        x = self.gelu(x)

        return x

    
# from learnable triangulation
class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_planes),
            nn.ReLU(),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm3d(out_planes)
            )
        self.last_relu = nn.ReLU()

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        out = self.last_relu(res + skip)
        return out


class MinicubesV2V(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.res1 = Res3DBlock(latent_dim, latent_dim)
        self.conv_out = nn.Conv3d(latent_dim, latent_dim,
                                  kernel_size=3, stride=1, padding=1)
        self.res2 = Res3DBlock(latent_dim, latent_dim)
        # self.skip1 = Res3DBlock(latent_dim, latent_dim)

        self._initialize_weights()
        
    def forward(self, x):
        # s1 = self.skip1(x)
        identity = x
        x = self.res1(x)
        x = self.res2(x)
        # x = x + s1
        x = self.conv_out(x + identity)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)













