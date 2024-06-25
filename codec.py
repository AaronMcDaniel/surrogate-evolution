import re

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as transforms

import primitives

# define list of backbones
BACKBONES = [
    "ConvNeXt",
    "DenseNet",
    "EfficientNet_V2",
    "Inception_V3",
    "MaxViT_T",
    "MobileNet_V3",
    "RegNet_X",
    "RegNet_Y",
    "ResNeXt",
    "ResNet",
    "ShuffleNet_V2",
    "Swin_V2",
    "ViT",
    "Wide_ResNet"
]


# these classes help extract features from existing classification models to obtain backbones
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def forward(self, x):
        raise NotImplementedError("FeatureExtractor is an abstract class.")

class ConvNeXtFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(ConvNeXtFeatures, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type}).features')

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class DenseNetFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(DenseNetFeatures, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type}).features')

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class EfficientNet_V2Features(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(EfficientNet_V2Features, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type}).features')

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class Inception_V3Features(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(Inception_V3Features, self).__init__()
        original_model = eval(f'torchvision.models.{backbone_name}(weights={weight_type})')
        self.model = nn.Sequential(
            original_model.Conv2d_1a_3x3,
            original_model.Conv2d_2a_3x3,
            original_model.Conv2d_2b_3x3,
            original_model.maxpool1,
            original_model.Conv2d_3b_1x1,
            original_model.Conv2d_4a_3x3,
            original_model.maxpool2,
            original_model.Mixed_5b,
            original_model.Mixed_5c,
            original_model.Mixed_5d,
            original_model.Mixed_6a,
            original_model.Mixed_6b,
            original_model.Mixed_6c,
            original_model.Mixed_6d,
            original_model.Mixed_6e,
            original_model.Mixed_7a,
            original_model.Mixed_7b,
            original_model.Mixed_7c,
        )

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class MaxViT_TFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(MaxViT_TFeatures, self).__init__()
        original_model = eval(f'torchvision.models.{backbone_name}(weights={weight_type})')
        self.model = nn.Sequential(
            original_model.stem,
            original_model.blocks
        )

    def forward(self, x):
        return self.model(x)

class MobileNet_V3Features(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(MobileNet_V3Features, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type}).features')

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class RegNet_XFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(RegNet_XFeatures, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type}).trunk_output')

    def forward(self, x):
        convLayer = nn.LazyConv2d(32, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class RegNet_YFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(RegNet_YFeatures, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type}).trunk_output')

    def forward(self, x):
        convLayer = nn.LazyConv2d(32, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class ResNeXtFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(ResNeXtFeatures, self).__init__()
        original_model = eval(f'torchvision.models.{backbone_name}(weights={weight_type})')
        self.model = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4
        )

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class ResNetFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(ResNetFeatures, self).__init__()
        original_model = eval(f'torchvision.models.{backbone_name}(weights={weight_type})')
        self.model = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4
        )

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class ShuffleNet_V2Features(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(ShuffleNet_V2Features, self).__init__()
        original_model = eval(f'torchvision.models.{backbone_name}(weights={weight_type})')
        self.model = nn.Sequential(
            original_model.conv1,  
            original_model.maxpool,
            original_model.stage2,
            original_model.stage3,
            original_model.stage4,
            original_model.conv5
        )

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)
    
class Swin_V2Features(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(Swin_V2Features, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type}).features')

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)

class ViTFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(ViTFeatures, self).__init__()
        self.model = eval(f'torchvision.models.{backbone_name}(weights={weight_type})')

    def forward(self, x):
        convLayer = nn.LazyConv2d(32, 1).to('cuda')
        x = convLayer(x)
        x = transforms.Resize((224,224))(x)
        return self.model(x)

class Wide_ResNetFeatures(FeatureExtractor):
    def __init__(self, backbone_name, weight_type):
        super(Wide_ResNetFeatures, self).__init__()
        original_model = eval(f'torchvision.models.{backbone_name}(weights={weight_type})')
        self.model = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4
        )

    def forward(self, x):
        convLayer = nn.LazyConv2d(3, 1).to('cuda')
        x = convLayer(x)
        return self.model(x)


# class representing output network that is dynamically built
class DynamicNetwork(nn.Module):
    def __init__(self, module_list, skips):
        super(DynamicNetwork, self).__init__()
        self.module_list = module_list
        self.skips = skips
        self.out_channels = None

    def forward(self, x):
        out = x
        for module in self.module_list:
            out = module(out)
        return out


# codec class
class Codec:
    def __init__(self, genome_encoding_strat, surrogate_encoding_strat, num_classes) -> None:
        self.genome_encoding_strat = genome_encoding_strat
        self.surrogate_encoding_strat = surrogate_encoding_strat
        self.num_classes = num_classes
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Codec using {self.device} device")


    def decode_genome(self, genome, num_loss_components):
        module_list = nn.ModuleList()
        if (self.genome_encoding_strat.lower() == 'tree'):
            # parse tree decoding into layer list
            expr = re.split(r'([(),])',genome)
            remove = [',', '']
            expr = [x for x in expr if x not in remove]
            stack = []
            idx = 0
            info = {}
            for element in expr:
                if element != ')':
                    stack.append(element)
                else:
                    arguments = []
                    while stack[-1] != '(':
                        arguments.insert(0, stack.pop())
                    stack.pop()
                    function = stack.pop()
                    try:
                        stack.append(str(eval(f'primitives.{function}({','.join(arguments)})')))
                    except: # this is where we add the layers
                        layer_info = [function]+[self.__parse_arg(x) for x in arguments]
                        info = self.add_to_module_list(module_list, idx, layer_info, num_loss_components)
                        idx += 1

            skip_info = torch.randn(4,4)
            model = self.add_head(module_list, skip_info)
            info['model'] = model
            return info
        

    def __parse_arg(self, s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s
    

    # builds module list from parsed information
    def add_to_module_list(self, module_list, idx, layer_info, num_loss_components):
        layer_name = layer_info[0]
        layer_args = layer_info[1:]

        if idx == 0 and 'IN0' in layer_args:
            layer_args.remove('IN0')

        # check if layer is an existing backbone
        if layer_name in BACKBONES:
            match layer_name:
                case "Inception_V3":
                    weightType = None if int(layer_args[0]) == 0 else '"IMAGENET1K_V1"'
                    backbone = Inception_V3Features(layer_name.lower(), weightType)
                    module_list.append(backbone)
                    return
                case "MaxViT_T":
                    weightType = None if int(layer_args[0]) == 0 else '"IMAGENET1K_V1"'
                    backbone = MaxViT_TFeatures(layer_name.lower(), weightType)
                    module_list.append(backbone)
                    return
                case "ConvNeXt":
                    sizeString = (list(primitives.ConvNeXtSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = ConvNeXtFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "DenseNet":
                    sizeString = (list(primitives.DenseNetSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = DenseNetFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "EfficientNet_V2":
                    sizeString = (list(primitives.EfficientNet_V2Size)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = EfficientNet_V2Features(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "MobileNet_V3":
                    sizeString = (list(primitives.MobileNet_V3Size)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = MobileNet_V3Features(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "RegNet_X":
                    sizeString = (list(primitives.RegNetSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = RegNet_XFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "RegNet_Y":
                    sizeString = (list(primitives.RegNetSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = RegNet_YFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "ResNeXt":
                    sizeString = (list(primitives.ResNeXtSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = ResNeXtFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "ResNet":
                    sizeString = (list(primitives.ResNetSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = ResNetFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "ShuffleNet_V2":
                    sizeString = (list(primitives.ShuffleNet_V2Size)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = ShuffleNet_V2Features(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "Swin_V2":
                    sizeString = (list(primitives.Swin_V2Size)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = Swin_V2Features(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "ViT":
                    sizeString = (list(primitives.ViTSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_SWAG_E2E_V1"'
                        case 2: weightType = '"IMAGENET1K_SWAG_LINEAR_V1"'
                    backbone = ViTFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return
                case "Wide_ResNet":
                    sizeString = (list(primitives.Wide_ResNetSize)[layer_args[0]]).name.lower()
                    if sizeString[0] == 'z':
                        backboneName = f'{layer_name.lower()}{sizeString[1:]}'
                    else:
                        backboneName = f'{layer_name.lower()}_{sizeString}'
                    weightType = None
                    match layer_args[1]:
                        case 0: weightType = None
                        case 1: weightType = '"IMAGENET1K_V1"'
                        case 2: weightType = '"IMAGENET1K_V2"'
                    backbone = Wide_ResNetFeatures(backboneName, weightType)
                    module_list.append(backbone)
                    return

        # check for special layers that require some processing        
        elif layer_name in ['LazyConv2d', 'LazyConvTranspose2d']:
            padding=(layer_args[5], layer_args[6])
            if min(layer_args[1], layer_args[2]) < 2*max(layer_args[5], layer_args[6]): # make sure that kernel size is less than twice the padding
                padding = (0,0)
            module_list.append(eval(f'nn.{layer_name.split('_')[0]}')(
                    out_channels=layer_args[0], 
                    kernel_size=(layer_args[1], layer_args[2]),
                    stride=(layer_args[3], layer_args[4]),
                    padding=padding,
                    padding_mode=(list(primitives.PaddingMode)[layer_args[7]]).name if layer_name == 'LazyConv2d' else 'zeros',
                    dilation=(layer_args[8], layer_args[9]),
                    groups=layer_args[10]
                ))
        
        elif layer_name == 'MaxPool2d':
            padding=(layer_args[4], layer_args[5])
            if min(layer_args[0], layer_args[1]) < 2*max(layer_args[4], layer_args[5]): # make sure that kernel size is less than twice the padding
                padding = (0,0)
            module_list.append(nn.MaxPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    stride=(layer_args[2], layer_args[3]),
                    padding=padding,
                    dilation=(layer_args[6], layer_args[7])
                ))
        
        elif layer_name == 'AvgPool2d':
            padding=(layer_args[4], layer_args[5])
            if min(layer_args[0], layer_args[1]) < 2*max(layer_args[4], layer_args[5]): # make sure that kernel size is less than twice the padding
                padding = (0,0)
            module_list.append(nn.AvgPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    stride=(layer_args[2], layer_args[3]),
                    padding=padding
                ))
            
        elif layer_name == 'FractionalMaxPool2d':
            x_outratio = layer_args[2] if layer_args[2] > 0.5 else 0.5
            y_outratio = layer_args[3] if layer_args[3] > 0.5 else 0.5
            module_list.append(nn.FractionalMaxPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    output_ratio=(x_outratio, y_outratio),
                ))
        
        elif layer_name == 'LPPool2d':
            module_list.append(nn.LPPool2d(
                    norm_type=layer_args[0],
                    kernel_size=(layer_args[1], layer_args[2]),
                    stride=(layer_args[3], layer_args[4]),
                ))
        
        elif layer_name in ['AdaptiveMaxPool2d', 'AdaptiveAvgPool2d']:
            module_list.append(eval(f'nn.{layer_name.split('_')[0]}')(
                    output_size=(layer_args[0], layer_args[1]),
                ))
        
        elif layer_name in ['Upsample_1D', 'Upsample_2D']:
            module_list.append(eval(f'nn.{layer_name.split('_')[0]}')(
                    scale_factor=layer_args[0],
                    mode=(list(primitives.UpsampleMode)[layer_args[1]]).name
                ))
            
        elif layer_name in ['Skip_1D', 'Skip_2D']:
            pass # TODO implement skip layer

        # detection head layer
        elif layer_name == 'Detection_Head':
            # extracting other details for training and val from the head
            loss_weights = layer_args[2:]
            if len(loss_weights) > num_loss_components:
                loss_weights = loss_weights[:num_loss_components]
            weights_sum = sum(loss_weights)
            loss_weights = [x/weights_sum for x in loss_weights]
            weight_tensor = torch.tensor(loss_weights, dtype=torch.float32)
            tensor = torch.zeros(num_loss_components, dtype=torch.float32)
            tensor[:len(weight_tensor)] = weight_tensor
            out_dict = {}
            optimizer_dict = eval(layer_args[0])
            scheduler_dict = eval(layer_args[1])
            out_dict['optimizer'] = optimizer_dict['optimizer']
            out_dict['lr_scheduler'] = scheduler_dict['lr_scheduler']
            for k, v in optimizer_dict.items():
                if k not in ['optimizer', 'eta_lower', 'eta_upper', 'step_lower', 'step_upper']:
                    out_dict[f'optimizer_{k}'] = v
            if optimizer_dict['optimizer'] == 'Rprop':
                out_dict[f'optimizer_etas'] = (optimizer_dict['eta_lower'], optimizer_dict['eta_upper'])
                out_dict[f'optimizer_step_sizes'] = (optimizer_dict['step_lower'], optimizer_dict['step_upper'])
            for k, v in scheduler_dict.items():
                if k != 'lr_scheduler':
                    out_dict[f'scheduler_{k}'] = v
            out_dict['loss_weights'] = tensor
            return out_dict
            
        else: # this is for layers that can have arguments simply unpacked
            module_list.append(eval(f'nn.{layer_name.split('_')[0]}')(*layer_args))
    

    # function to add appropriate detection head to custom backbone
    def add_head(self, module_list, skip_info): # adds rcnn head for now should be customizable later
        dummy_input = torch.randn(1, 3, 2048, 2448).to(self.device)
        model = DynamicNetwork(module_list, skip_info)
        test_model = model.to(self.device)
        output = test_model(dummy_input)
        model.out_channels = output.shape[1]
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=21,
            sampling_ratio=4
        )
        model = FasterRCNN(
            model,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
        return model   
        