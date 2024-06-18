import re

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import primitives


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


    def decode_genome(self, genome):
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
                    arguments = nn.ModuleList()
                    while stack[-1] != '(':
                        arguments.insert(0, stack.pop())
                    stack.pop()
                    function = stack.pop()
                    try:
                        stack.append(str(eval(f'primitives.{function}({','.join(arguments)})')))
                    except: # this is where we add the layers
                        layer_info = [function]+[self.__parse_arg(x) for x in arguments]
                        info = self.add_to_module_list(module_list, idx, layer_info)
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
    

    def add_to_module_list(self, module_list, idx, layer_info):
        layer_name = layer_info[0]
        layer_args = layer_info[1:]

        if idx == 0 and 'IN0' in layer_args:
            layer_args.remove('IN0')

        if layer_name in ['LazyConv2d', 'LazyConvTranspose2d']:
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
            module_list.append(nn.FractionalMaxPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    output_ratio=(layer_args[2], layer_args[3]),
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

        elif layer_name == 'Detection_Head':
            loss_weights = layer_args[4:]
            weights_sum = sum(loss_weights)
            loss_weights = [x/weights_sum for x in loss_weights]
            return {
                'optimizer': (list(primitives.Optimizer)[layer_args[0]]).name,
                'lr': layer_args[1],
                'iou_thresh': layer_args[2],
                'conf_thresh': layer_args[3],
                'loss_weights': {
                    'iou': loss_weights[0],
                    'diou': loss_weights[1],
                    'giou': loss_weights[2],
                    'ciou': loss_weights[3],
                    'precision': loss_weights[4],
                    'recall': loss_weights[5],
                    'ap': loss_weights[6],
                    'l2_norm_center': loss_weights[7],
                    'l2_norm_area': loss_weights[8]
                }
            }
            
        else: # this is for layers that can have arguments simply unpacked
            module_list.append(eval(f'nn.{layer_name.split('_')[0]}')(*layer_args))
    

    def add_head(self, module_list, skip_info): # adds rcnn head for now should be customizable later
        dummy_input = torch.randn(1, 3, 2048, 2448).to(self.device)
        model = DynamicNetwork(module_list, skip_info)
        test_model = model.to(self.device)
        output = test_model(dummy_input)
        model.out_channels = output.shape[1]
        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128),),
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
        