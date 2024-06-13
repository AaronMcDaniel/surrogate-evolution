import re
import torch
import torch.nn as nn
from torchsummary import summary
import primitives


class DynamicNetwork(nn.Module):
    def __init__(self, module_list, skips):
        super(DynamicNetwork, self).__init__()
        self.module_list = module_list
        self.skips = skips

    def forward(self, x):
        out = x
        for module in self.module_list:
            out = module(out)
        return out


class Codec:
    def __init__(self, genome_encoding_strat, surrogate_encoding_strat) -> None:
        self.genome_encoding_strat = genome_encoding_strat
        self.surrogate_encoding_strat = surrogate_encoding_strat
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
                        self.add_to_module_list(module_list, idx, layer_info)
                        idx += 1

                        # print(f'{function}({','.join(arguments)})')
            print(module_list)
            skip_info = torch.randn(4,4)
            model = DynamicNetwork(module_list, skip_info)
            return model
        
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
            # no need to add backbone cell, input is already image tensor
            layer_args.remove('IN0')
        elif idx == 0 and not 'IN0' in layer_args:
            # add custom backbone here for now will just add a flatten
            module_list.append(nn.Flatten())

        if layer_name in ['LazyConv2d', 'LazyConvTranspose2d']:
            module_list.append(eval(f'nn.{layer_name.split('_')[0]}')(
                    out_channels=layer_args[0], 
                    kernel_size=(layer_args[1], layer_args[2]),
                    stride=(layer_args[3], layer_args[4]),
                    padding=(layer_args[5], layer_args[6]),
                    padding_mode=(list(primitives.PaddingMode)[layer_args[7]]).name,
                    dilation=(layer_args[8], layer_args[9]),
                    groups=layer_args[10] 
                ))
        
        elif layer_name == 'MaxPool2d':
            module_list.append(nn.MaxPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    stride=(layer_args[2], layer_args[3]),
                    padding=(layer_args[4], layer_args[5]),
                    dilation=(layer_args[6], layer_args[7])
                ))
        
        elif layer_name == 'AvgPool2d':
            module_list.append(nn.AvgPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    stride=(layer_args[2], layer_args[3]),
                    padding=(layer_args[4], layer_args[5])
                ))
            
        elif layer_name == 'FractionalMaxPool2d':
            module_list.append(nn.FractionalMaxPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    output_size=(layer_args[2], layer_args[3]),
                ))
        
        elif layer_name == 'LPPool2d':
            module_list.append(nn.LPPool2d(
                    kernel_size=(layer_args[0], layer_args[1]),
                    stride=(layer_args[2], layer_args[3]),
                    padding=(layer_args[4], layer_args[5])
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
            # detection head goes here for now just gonna put a linear layer
            module_list.append(nn.LazyLinear(4))
            
        else: # this is for layers that can have arguments simply unpacked
            module_list.append(eval(f'nn.{layer_name.split('_')[0]}')(*layer_args))



test_str = "Detection_Head(ReLU_1D(Flatten(LazyConvTranspose2d(LazyConv2d(IN0, 55, 7, 8, 6, 5, 1, 1, 0, 3, 10, 1), protectedDiv(32, 10), kernelMul(3, 2), protectedSub(4, 6), strideMul(7, 3), protectedSub(7, 8), protectedSub(5, 9), protectedDiv(8, 6), dummyOp(0), dilationAdd(9, 5), 2, 1)), floatAdd(floatAdd(protectedDiv(0.9649450870627716, 0.713984394386917), floatAdd(0.48838292526077265, 0.45496959791456104)), floatAdd(floatAdd(0.49007733856716007, 0.344404196807212), 0.1481169852772074))))"

codec = Codec('Tree', '')
model = codec.decode_genome(test_str)
model = model.to(codec.device)
dummy_input = torch.randn(1, 3, 2048, 2448).to(codec.device)
output = model(dummy_input)
summary(model, input_size=(3, 2048, 2448))