from enum import Enum
from functools import partial
import itertools
import math
import os
import random

from deap import gp
import toml

# loading config file
configs = toml.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf.toml"))
pipeline_config = configs["pipeline"]
codec_config = configs["codec"]
num_loss_components = int(codec_config['num_loss_components'])


# max bounds for layers
MAX_CHANNEL_SIZE = 64
MAX_KERNEL_SIZE = 9
MAX_STRIDE_SIZE = 9
MAX_PADDING_SIZE = 9
MAX_OUTPUT_SIZE = 3000
MAX_DILATION_SIZE = 10
MAX_GROUP_SIZE = 1
MAX_SKIP_SIZE = 5
MAX_PNORM_SIZE = 3


# placeholder classes to act as types for DEAP's strongly typed primitive set
class Tensor3D:
    pass

class FinalTensor: # acts as an end type
    pass


# separate wrapper classes for different hyperparameters so bounds can be placed on individual ephemeral constants and crossovers are valid
class ChannelSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class KernelSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class StrideSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class PaddingSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class OutputSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class DilationSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class GroupSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class SkipSize(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class PNorm(float):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class ProbFloat(float):
    def __init__(self, num) -> None:
        super().__init__()
    pass

class GenericInt(int): # exists so that anything with GenericInt isn't treated as a superclass of other int-inheriting types
    def __init__(self, num) -> None:
        super().__init__()
    pass

class Optimizer(dict):
    def __init__(self, initial_dict=None, **kwargs):
        if initial_dict is None:
            initial_dict = {}
        super(Optimizer, self).__init__(initial_dict)
        self.update(kwargs)

    def __str__(self):
        return super().__str__()
    
class Scheduler(dict):
    def __init__(self, initial_dict=None, **kwargs):
        if initial_dict is None:
            initial_dict = {}
        super(Scheduler, self).__init__(initial_dict)
        self.update(kwargs)

    def __str__(self):
        return super().__str__()


# input parameters that are enums
class PaddingMode(Enum):
    zeros = 0,
    reflect = 1,
    replicate = 2,
    circular = 3

class UpsampleMode(Enum):
    nearest = 0,
    linear = 1,
    bilinear = 2,
    bicubic = 3,
    trilinear = 4

class SkipMergeType(Enum):
    concat = 0,
    add = 1,

class ConvNeXtSize(Enum):
    Base = 0,
    Large = 1,
    Small = 2,
    Tiny = 3

class DenseNetSize(Enum):
    z121 = 0,
    z161 = 1,
    z169 = 2,
    z201 = 3

class EfficientNet_V2Size(Enum):
    L = 0,
    M = 1,
    S = 2

class MobileNet_V3Size(Enum):
    Large = 0,
    Small = 1

class RegNetSize(Enum):
    z_16GF = 0,
    z_1_6GF = 1,
    z_32GF = 2,
    z_3_2GF = 3,
    z_400MF = 4,
    z_800MF = 5,
    z_8GF = 6,

class ResNeXtSize(Enum):
    z101_32X8D = 0,
    z50_32X4D = 1

class ResNetSize(Enum):
    z101 = 0,
    z152 = 1,
    z50 = 2

class ShuffleNet_V2Size(Enum):
    X0_5 = 0,
    X1_0 = 1,
    X1_5 = 2,
    X2_0 = 3

class Swin_V2Size(Enum):
    B = 0,
    S = 1,
    T = 2

class ViTSize(Enum):
    B_16 = 0,
    H_14 = 1,
    L_16 = 2

class Wide_ResNetSize(Enum):
    z101_2 = 0,
    z50_2 = 1

class Weights(Enum):
    WEIGHT0 = 0,
    WEIGHT1 = 1,
    WEIGHT2 = 2

class BoolWeight(Enum):
    WEIGHTFALSE = 0,
    WEIGHTTRUE = 1

class AnnealStrategy(Enum):
    cos = 0
    linear = 1

class CyclicLRMode(Enum):
    triangular = 0
    triangular2 = 1
    exp_range = 2

class CyclicLRScaleMode(Enum):
    cycle = 0
    iterations = 1


# Conv layers
def LazyConv2d(tensor: Tensor3D, out_channels: ChannelSize, kernel_size0: KernelSize, kernel_size1: KernelSize, stride0: StrideSize, stride1: StrideSize, padding0: PaddingSize, padding1: PaddingSize, padding_mode: PaddingMode, 
               dilation0: DilationSize, dilation1:  DilationSize, groups: GroupSize): # assume bias is always true
    return Tensor3D()

def LazyConvTranspose2d(tensor: Tensor3D, out_channels: ChannelSize, kernel_size0: KernelSize, kernel_size1: KernelSize, stride0: StrideSize, stride1: StrideSize, padding0: PaddingSize, padding1: PaddingSize, padding_mode: PaddingMode, 
               dilation0: DilationSize, dilation1:  DilationSize, groups: GroupSize):
    return Tensor3D()


# Pool layers
def MaxPool2d(tensor: Tensor3D, kernel_size0: KernelSize, kernel_size1: KernelSize, stride0: StrideSize, stride1: StrideSize, padding0: PaddingSize, padding1: PaddingSize, dilation0: DilationSize, dilation1:  DilationSize):
    return Tensor3D()

def AvgPool2d(tensor: Tensor3D, kernel_size0: KernelSize, kernel_size1: KernelSize, stride0: StrideSize, stride1: StrideSize, padding0: PaddingSize, padding1: PaddingSize):
    return Tensor3D()

def FractionalMaxPool2d(tensor: Tensor3D, kernel_size0: KernelSize, kernel_size1: KernelSize, output_ratio0: ProbFloat, output_ratio1: ProbFloat):
    return Tensor3D()

def LPPool2d(tensor: Tensor3D, norm_type: PNorm, kernel_size0 : KernelSize, kernel_size1: KernelSize, stride0: StrideSize, stride1: StrideSize):
    return Tensor3D()

def AdaptiveMaxPool2d(tensor: Tensor3D, output_size0: OutputSize, output_size1: OutputSize):
    return Tensor3D()

def AdaptiveAvgPool2d(tensor: Tensor3D, output_size0: OutputSize, output_size1: OutputSize):
    return Tensor3D()


# TODO: Padding Layers


# Activation layers
def ReLU_2D(tensor: Tensor3D):
    return Tensor3D()

def LeakyReLU_2D(tensor: Tensor3D, negative_slope: float):
    return Tensor3D()

def RReLU_2D(tensor: Tensor3D, lower: float, upper: float):
    return Tensor3D()

def LogSigmoid_2D(tensor: Tensor3D):
    return Tensor3D()

def Sigmoid_2D(tensor: Tensor3D):
    return Tensor3D()

def Tanh_2D(tensor: Tensor3D):
    return Tensor3D()

def Threshold_2D(tensor: Tensor3D, threshold: float, value: float):
    return Tensor3D()


# TODO: MultiHeadAttention


# Normalization and Dropout layers
def LazyBatchNorm2d(tensor: Tensor3D, eps: float, momentum: float):
    return Tensor3D()

def Dropout_2D(tensor: Tensor3D, p: ProbFloat):
    return Tensor3D()


# TODO: transformer layers


# Vision layers
def Upsample_2D(tensor: Tensor3D, scaling_factor: float, mode: UpsampleMode):
    return Tensor3D()


# Skip Connection Support: The way this is supposed to work is that a Skip layer can be added anywhere and the skip_by parameter tells us how many layers to skip
# by which allows us to then use the chosen merge_type to merge after skipping. It is likely that the merging will not be straighforward since dimensions may vary
# and the merge location may not even exist if a bad skip_by value is chosen, but we can either heal when decoding or hope the GA will figure out how to use padding 
# layers or similar to make it work.  
def Skip_2D(tensor: Tensor3D, skip_by: SkipSize, merge_type: SkipMergeType):
    return Tensor3D()

# Detection Heads
def FasterRCNN_Head(tensor: Tensor3D, optimizer: Optimizer, scheduler: Scheduler):
    return FinalTensor()

def FCOS_Head(tensor: Tensor3D, optimizer: Optimizer, scheduler: Scheduler):
    return FinalTensor()

def RetinaNet_Head(tensor: Tensor3D, optimizer: Optimizer, scheduler: Scheduler):
    return FinalTensor()

def SSD_Head(tensor: Tensor3D, optimizer: Optimizer, scheduler: Scheduler):
    return FinalTensor()


# Backbones
def ConvNeXt(tensor: Tensor3D, convnextsize: ConvNeXtSize, weights: BoolWeight):
    return Tensor3D()

def DenseNet(tensor: Tensor3D, densenetsize: DenseNetSize, weights: BoolWeight):
    return Tensor3D()

def EfficientNet_V2(tensor: Tensor3D, efficientnetsize: EfficientNet_V2Size, weights: BoolWeight):
    return Tensor3D()

def Inception_V3(tensor: Tensor3D, weights: BoolWeight):
    return Tensor3D()

def MaxViT_T(tensor: Tensor3D, weights: BoolWeight):
    return Tensor3D()

def MobileNet_V3(tensor: Tensor3D, mobilenetsize: MobileNet_V3Size, weights: BoolWeight):
    return Tensor3D()

def RegNet_X(tensor: Tensor3D, regnetsize: RegNetSize, weights: Weights):
    return Tensor3D()

def RegNet_Y(tensor: Tensor3D, regnetsize: RegNetSize, weights: Weights):
    return Tensor3D()

def ResNeXt(tensor: Tensor3D, resnextsize: ResNeXtSize, weights: Weights):
    return Tensor3D()

def ResNet(tensor: Tensor3D, resnetsize: ResNetSize, weights: Weights):
    return Tensor3D()

def ShuffleNet_V2(tensor: Tensor3D, shufflenetsize: ShuffleNet_V2Size, weights: BoolWeight):
    return Tensor3D()

def Swin_V2(tensor: Tensor3D, swinsize: Swin_V2Size, weights: BoolWeight):
    return Tensor3D()

def ViT(tensor: Tensor3D, vitsize: ViTSize, weights: Weights):
    return Tensor3D()

def Wide_ResNet(tensor: Tensor3D, wideresnetsize: Wide_ResNetSize, weights: Weights):
    return Tensor3D()


# Optimizers
def SGD(lr: float, momentum: float, weight_decay: float, dampening: float) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-1)
    momentum = transform_value(momentum, 0, 0.9)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    dampening = transform_value(dampening, 0, 0.9)
    return Optimizer({'optimizer': 'SGD', 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay, 'dampening': dampening})

def Adadelta(lr: float, rho: float, weight_decay: float) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-1)
    rho = transform_value(rho, 0.9, 0.999)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'Adadelta', 'lr': lr, 'rho': rho, 'weight_decay': weight_decay})

def Adagrad(lr: float, weight_decay: float) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-1)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'Adagrad', 'lr': lr, 'weight_decay': weight_decay})

def Adam(lr: float, weight_decay: float, amsgrad: bool) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-3)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'Adam', 'lr': lr, 'weight_decay': weight_decay, 'amsgrad': amsgrad})

def AdamW(lr: float, weight_decay: float, amsgrad: bool) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-3)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'AdamW', 'lr': lr, 'weight_decay': weight_decay, 'amsgrad': amsgrad})

def Adamax(lr: float, weight_decay: float) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-3)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'Adamax', 'lr': lr, 'weight_decay': weight_decay})

def ASGD(lr: float, lambd: float, alpha: float, t0: float, weight_decay: float) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-1)
    lambd = transform_value(lambd, 1e-5, 1e-1)
    alpha = transform_value(alpha, 1e-5, 1e-1)
    t0 = transform_value(t0, 1, 1e3)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'ASGD', 'lr': lr, 'lambd': lambd, 'alpha': alpha, 't0': t0, 'weight_decay': weight_decay})

def NAdam(lr: float, weight_decay: float, momentum_decay: float, decoupled_weight_decay: bool) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-3)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    momentum_decay = transform_value(momentum_decay, 0.9, 0.999)
    return Optimizer({'optimizer': 'NAdam', 'lr': lr, 'weight_decay': weight_decay, 'momentum_decay': momentum_decay, 'decoupled_weight_decay': decoupled_weight_decay})

def RAdam(lr: float, weight_decay: float, decoupled_weight_decay: bool) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-3)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'RAdam', 'lr': lr, 'weight_decay': weight_decay, 'decoupled_weight_decay': decoupled_weight_decay})

def RMSprop(lr: float, momentum: float, alpha: float, centered: bool, weight_decay: float) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-1)
    momentum = transform_value(momentum, 0, 0.9)
    alpha = transform_value(alpha, 0.9, 0.999)
    weight_decay = transform_value(weight_decay, 0, 1e-2)
    return Optimizer({'optimizer': 'RMSprop', 'lr': lr, 'momentum': momentum, 'alpha': alpha, 'centered': centered, 'weight_decay': weight_decay})

def Rprop(lr: float, eta_lower: float, eta_upper: float, step_lower: float, step_upper: float) -> Optimizer:
    lr = transform_value(lr, 1e-4, 1e-1)
    eta_lower = transform_value(eta_lower, 1e-6, 1e-2)
    eta_upper = transform_value(eta_upper, 1e-3, 1)
    step_lower = transform_value(step_lower, 1e-5, 0.1)
    step_upper = transform_value(step_upper, 1, 100)
    return Optimizer({'optimizer': 'Rprop', 'lr': lr, 'eta_lower': eta_lower, 'eta_upper': eta_upper, 'step_lower': step_lower, 'step_upper': step_upper})


# Schedulers
def StepLR(step_size: int, gamma: float) -> Scheduler:
    step_size = int(transform_value(step_size, 1, 100))
    gamma = transform_value(gamma, 0.01, 0.99)
    return Scheduler({'lr_scheduler': 'StepLR', 'step_size': step_size, 'gamma': gamma})

def MultiStepLR(gamma: float) -> Scheduler:
    gamma = transform_value(gamma, 0.01, 0.99)
    return Scheduler({'lr_scheduler': 'MultiStepLR', 'gamma': gamma})

def ExponentialLR(gamma: float) -> Scheduler:
    gamma = transform_value(gamma, 0.01, 0.99)
    return Scheduler({'lr_scheduler': 'ExponentialLR', 'gamma': gamma})

def ReduceLROnPlateau(factor: float, patience: int, threshold: float, cooldown: int, min_lr: float, eps: float) -> Scheduler:
    factor = transform_value(factor, 0.01, 0.99)
    patience = int(transform_value(patience, 1, 100))
    threshold = transform_value(threshold, 1e-5, 1e-2)
    cooldown = int(transform_value(cooldown, 0, 10))
    min_lr = transform_value(min_lr, 0, 1e-2)
    eps = transform_value(eps, 1e-8, 1e-4)
    return Scheduler({'lr_scheduler': 'ReduceLROnPlateau', 'factor': factor, 'patience': patience, 'threshold': threshold, 'cooldown': cooldown, 'min_lr': min_lr, 'eps': eps})

def CosineAnnealingLR(T_max: int, eta_min: float) -> Scheduler:
    T_max = int(transform_value(T_max, 1, 100))
    eta_min = transform_value(eta_min, 0, 1e-2)
    return Scheduler({'lr_scheduler': 'CosineAnnealingLR', 'T_max': T_max, 'eta_min': eta_min})

def CosineAnnealingWarmRestarts(T_0: int, T_mult: int, eta_min: float) -> Scheduler:
    T_0 = int(transform_value(T_0, 1, 100))
    T_mult = int(transform_value(T_mult, 1, 10))
    eta_min = transform_value(eta_min, 0, 1e-2)
    return Scheduler({'lr_scheduler': 'CosineAnnealingWarmRestarts', 'T_0': T_0, 'T_mult': T_mult, 'eta_min': eta_min})

def OneCycleLR(max_lr: float, pct_start: float, anneal_strategy: AnnealStrategy, cycle_momentum: bool, base_momentum: float, max_momentum: float, div_factor: float, final_div_factor: float, three_phase: bool) -> Scheduler:
    max_lr = transform_value(max_lr, 1e-3, 1)
    pct_start = transform_value(pct_start, 0, 1)
    base_momentum = transform_value(base_momentum, 0.5, 1)
    max_momentum = transform_value(max_momentum, 0.5, 1)
    div_factor = transform_value(div_factor, 1, 100)
    final_div_factor = transform_value(final_div_factor, 1, 1e5)
    return Scheduler({'lr_scheduler': 'OneCycleLR', 'max_lr': max_lr, 'pct_start': pct_start, 'anneal_strategy': list(AnnealStrategy)[anneal_strategy].name, 'cycle_momentum': cycle_momentum, 'base_momentum': base_momentum, 'max_momentum': max_momentum, 'div_factor': div_factor, 'final_div_factor': final_div_factor, 'three_phase': three_phase})

def ConstantLR(factor: float, total_iters: int) -> Scheduler:
    factor = transform_value(factor, 0.01, 1)
    total_iters = int(transform_value(total_iters, 1, 100))
    return Scheduler({'lr_scheduler': 'ConstantLR', 'factor': factor, 'total_iters': total_iters})

def LinearLR(start_factor: float, end_factor: float, total_iters: int) -> Scheduler:
    start_factor = transform_value(start_factor, 0.01, 1)
    end_factor = transform_value(end_factor, 0.01, 1)
    total_iters = int(transform_value(total_iters, 1, 100))
    return Scheduler({'lr_scheduler': 'LinearLR', 'start_factor': start_factor, 'end_factor': end_factor, 'total_iters': total_iters})

def PolynomialLR(max_lr: float, total_steps: int, power: float) -> Scheduler:
    max_lr = transform_value(max_lr, 1e-4, 1)
    total_steps = int(transform_value(total_steps, 1, 1000))
    power = transform_value(power, 0.5, 3)
    return Scheduler({'lr_scheduler': 'PolynomialLR', 'max_lr': max_lr, 'total_steps': total_steps, 'power': power})

def CyclicLR(base_lr: float, max_lr: float, step_size_up: int, step_size_down: int, mode: CyclicLRMode, gamma: float, scale_mode: CyclicLRScaleMode, cycle_momentum: bool, base_momentum: float, max_momentum: float) -> Scheduler:
    base_lr = transform_value(base_lr, 1e-4, 1)
    max_lr = transform_value(max_lr, base_lr, 1)
    step_size_up = int(transform_value(step_size_up, 1000, 10000))
    step_size_down = int(transform_value(step_size_down, 1000, 10000))
    gamma = transform_value(gamma, 0.5, 1)
    base_momentum = transform_value(base_momentum, 0.5, 1)
    max_momentum = transform_value(max_momentum, 0.5, 1)
    return Scheduler({'lr_scheduler': 'CyclicLR', 'base_lr': base_lr, 'max_lr': max_lr, 'step_size_up': step_size_up, 'step_size_down': step_size_down, 'mode': list(CyclicLRMode)[mode].name, 'gamma': gamma, 'scale_mode': list(CyclicLRScaleMode)[scale_mode].name, 'cycle_momentum': cycle_momentum, 'base_momentum': base_momentum, 'max_momentum': max_momentum})


# creating primitive set from layers and components
pset = gp.PrimitiveSetTyped("MAIN", [Tensor3D], FinalTensor, "IN")
pset.addPrimitive(LazyConv2d, 
                  [Tensor3D, ChannelSize, KernelSize, KernelSize, StrideSize, StrideSize, PaddingSize, PaddingSize, PaddingMode, DilationSize, DilationSize, GroupSize], 
                  Tensor3D)

pset.addPrimitive(LazyConvTranspose2d,
                  [Tensor3D, ChannelSize, KernelSize, KernelSize, StrideSize, StrideSize, PaddingSize, PaddingSize, PaddingMode, DilationSize, DilationSize, GroupSize],
                  Tensor3D)

pset.addPrimitive(MaxPool2d,
                  [Tensor3D, KernelSize, KernelSize, StrideSize, StrideSize, PaddingSize, PaddingSize, DilationSize, DilationSize],
                  Tensor3D)

pset.addPrimitive(AvgPool2d,
                  [Tensor3D, KernelSize, KernelSize, StrideSize, StrideSize, PaddingSize, PaddingSize],
                  Tensor3D)

pset.addPrimitive(FractionalMaxPool2d,
                  [Tensor3D, KernelSize, KernelSize, ProbFloat, ProbFloat],
                  Tensor3D)

pset.addPrimitive(LPPool2d,
                  [Tensor3D, PNorm, KernelSize, KernelSize, StrideSize, StrideSize],
                  Tensor3D)

pset.addPrimitive(AdaptiveMaxPool2d,
                  [Tensor3D, OutputSize, OutputSize],
                  Tensor3D)

pset.addPrimitive(AdaptiveAvgPool2d,
                  [Tensor3D, OutputSize, OutputSize],
                  Tensor3D)

pset.addPrimitive(ReLU_2D,
                  [Tensor3D],
                  Tensor3D)

pset.addPrimitive(LeakyReLU_2D,
                  [Tensor3D, float],
                  Tensor3D)

pset.addPrimitive(LogSigmoid_2D,
                  [Tensor3D],
                  Tensor3D)

pset.addPrimitive(Sigmoid_2D,
                  [Tensor3D],
                  Tensor3D)

pset.addPrimitive(Tanh_2D,
                  [Tensor3D],
                  Tensor3D)

pset.addPrimitive(Threshold_2D,
                  [Tensor3D, float, float],
                  Tensor3D)

pset.addPrimitive(LazyBatchNorm2d,
                  [Tensor3D, float, float],
                  Tensor3D)

pset.addPrimitive(Dropout_2D,
                  [Tensor3D, ProbFloat],
                  Tensor3D)

pset.addPrimitive(Skip_2D,
                  [Tensor3D, SkipSize, SkipMergeType],
                  Tensor3D)

pset.addPrimitive(FasterRCNN_Head,
                  [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(float, num_loss_components)),
                  FinalTensor)

pset.addPrimitive(FCOS_Head,
                  [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(float, num_loss_components)),
                  FinalTensor)

pset.addPrimitive(RetinaNet_Head,
                  [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(float, num_loss_components)),
                  FinalTensor)

# pset.addPrimitive(SSD_Head,
#                   [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(float, num_loss_components)),
#                   FinalTensor)

pset.addPrimitive(Upsample_2D,
                  [Tensor3D, float, UpsampleMode],
                  Tensor3D)

pset.addPrimitive(ConvNeXt,
                  [Tensor3D, ConvNeXtSize, BoolWeight],
                  Tensor3D)

pset.addPrimitive(DenseNet,
                  [Tensor3D, DenseNetSize, BoolWeight],
                  Tensor3D)

pset.addPrimitive(EfficientNet_V2,
                  [Tensor3D, EfficientNet_V2Size, BoolWeight],
                  Tensor3D)

pset.addPrimitive(Inception_V3,
                  [Tensor3D, BoolWeight],
                  Tensor3D)

# pset.addPrimitive(MaxViT_T,
#                   [Tensor3D, BoolWeight],
#                   Tensor3D)

pset.addPrimitive(MobileNet_V3,
                  [Tensor3D, MobileNet_V3Size, BoolWeight],
                  Tensor3D)

pset.addPrimitive(RegNet_X,
                  [Tensor3D, RegNetSize, Weights],
                  Tensor3D)

pset.addPrimitive(RegNet_Y,
                  [Tensor3D, RegNetSize, Weights],
                  Tensor3D)

pset.addPrimitive(ResNeXt,
                  [Tensor3D, ResNeXtSize, Weights],
                  Tensor3D)

pset.addPrimitive(ResNet,
                  [Tensor3D, ResNetSize, Weights],
                  Tensor3D)

pset.addPrimitive(ShuffleNet_V2,
                  [Tensor3D, ShuffleNet_V2Size, BoolWeight],
                  Tensor3D)

pset.addPrimitive(Swin_V2,
                  [Tensor3D, Swin_V2Size, BoolWeight],
                  Tensor3D)

# pset.addPrimitive(ViT,
#                   [Tensor3D, ViTSize, Weights],
#                   Tensor3D)

pset.addPrimitive(Wide_ResNet,
                  [Tensor3D, Wide_ResNetSize, Weights],
                  Tensor3D)

pset.addPrimitive(SGD,
                  [float, float, float, float],
                  Optimizer)

pset.addPrimitive(Adadelta,
                  [float, float, float],
                  Optimizer)

pset.addPrimitive(Adagrad,
                  [float, float],
                  Optimizer)

pset.addPrimitive(Adam,
                  [float, float, bool],
                  Optimizer)

pset.addPrimitive(AdamW,
                  [float, float, bool],
                  Optimizer)

pset.addPrimitive(Adamax,
                  [float, float],
                  Optimizer)

pset.addPrimitive(ASGD,
                  [float, float, float, float, float],
                  Optimizer)

pset.addPrimitive(NAdam,
                  [float, float, float, bool],
                  Optimizer)

pset.addPrimitive(RAdam,
                  [float, float, bool],
                  Optimizer)

pset.addPrimitive(RMSprop,
                  [float, float, float, bool, float],
                  Optimizer)

pset.addPrimitive(Rprop,
                  [float, float, float, float, float],
                  Optimizer)

pset.addPrimitive(StepLR,
                  [int, float],
                  Scheduler)

pset.addPrimitive(MultiStepLR,
                  [float],
                  Scheduler)

pset.addPrimitive(ExponentialLR,
                  [float],
                  Scheduler)

pset.addPrimitive(ReduceLROnPlateau,
                  [float, int, float, int, float, float],
                  Scheduler)

pset.addPrimitive(CosineAnnealingLR,
                  [int, float],
                  Scheduler)

pset.addPrimitive(CosineAnnealingWarmRestarts,
                  [int, int, float],
                  Scheduler)

pset.addPrimitive(OneCycleLR,
                  [float, float, AnnealStrategy, bool, float, float, float, float, bool],
                  Scheduler)

pset.addPrimitive(ConstantLR,
                  [float, int],
                  Scheduler)

pset.addPrimitive(LinearLR,
                  [float, float, int],
                  Scheduler)

pset.addPrimitive(PolynomialLR,
                  [float, int, float],
                  Scheduler)

pset.addPrimitive(CyclicLR,
                  [float, float, int, int, CyclicLRMode, float, CyclicLRScaleMode, bool, float, float],
                  Scheduler)


# Basic operators
def add(a, b):
    return a+b

def protectedSub(a, b):
    return abs(a-b)

def mul(a, b):
    return a*b

def protectedDiv(left, right):
    if (isinstance(left, int) and isinstance(right, int)):
        try: return left // right
        except ZeroDivisionError: return 1
    else:
        try: return left / right
        except ZeroDivisionError: return 1


# Datatype conversions
def toChannel(a):
    if a == 0:
        return 1
    return ChannelSize(a%MAX_CHANNEL_SIZE)+1 if a > MAX_CHANNEL_SIZE else ChannelSize(a)

def toKernel(a):
    if a < 3: return 3
    return KernelSize(a%MAX_KERNEL_SIZE)+3 if a > MAX_KERNEL_SIZE else KernelSize(a)

def toStride(a):
    if a == 0:
        return 1
    return StrideSize(a%MAX_STRIDE_SIZE)+1 if a > MAX_STRIDE_SIZE else StrideSize(a)

def toPadding(a):
    return PaddingSize(a%MAX_PADDING_SIZE) if a > MAX_PADDING_SIZE else PaddingSize(a)

def toOutput(a):
    if a < 7:
        return 7
    return OutputSize(a%MAX_OUTPUT_SIZE)+7 if a > MAX_OUTPUT_SIZE else OutputSize(a)

def toDilation(a):
    if a == 0:
        return 1
    return DilationSize(a%MAX_DILATION_SIZE)+1 if a > MAX_DILATION_SIZE else DilationSize(a)

def toGroup(a):
    if a == 0:
        return 1
    return GroupSize(a%MAX_GROUP_SIZE)+1 if a > MAX_GROUP_SIZE else GroupSize(a)

def toSkip(a):
    if a == 0:
        return 1
    return SkipSize(a%MAX_SKIP_SIZE)+1 if a > MAX_SKIP_SIZE else SkipSize(a)

def toPNorm(a):
    if a < 2:
        return 2
    return PNorm(a%MAX_PNORM_SIZE)+2 if a > MAX_PNORM_SIZE else PNorm(a)

def toProbFloat(a):
    return a%1

def dummyOp(input):
    return input


# helper method to transform values
def transform_value(value, lower_bound, upper_bound):
    # Apply the exponential decay function
    transformed = math.exp(-(value+1))
    # Scale the transformed value to the provided bounds
    scaled_value = lower_bound + (upper_bound - lower_bound) * transformed
    return scaled_value


# helper to generate a random boolean
def genRandBool():
    return bool(random.getrandbits(1))


# adding functions as primitives
pset.addPrimitive(add, [GenericInt, GenericInt], GenericInt)
pset.addPrimitive(add, [float, float], float)
pset.addPrimitive(protectedSub, [GenericInt, GenericInt], GenericInt)
pset.addPrimitive(protectedSub, [float, float], float)
pset.addPrimitive(mul, [GenericInt, GenericInt], GenericInt)
pset.addPrimitive(mul, [float, float], float)
pset.addPrimitive(protectedDiv, [GenericInt, GenericInt], GenericInt)
pset.addPrimitive(protectedDiv, [float, float], float)
pset.addPrimitive(toChannel, [GenericInt], ChannelSize)
pset.addPrimitive(toKernel, [GenericInt], KernelSize)
pset.addPrimitive(toStride, [GenericInt], StrideSize)
pset.addPrimitive(toPadding, [GenericInt], PaddingSize)
pset.addPrimitive(toOutput, [GenericInt], OutputSize)
pset.addPrimitive(toDilation, [GenericInt], DilationSize)
pset.addPrimitive(toGroup, [GenericInt], GroupSize)
pset.addPrimitive(toSkip, [GenericInt], SkipSize)
pset.addPrimitive(toPNorm, [float], PNorm)
pset.addPrimitive(toProbFloat, [float], ProbFloat)
pset.addPrimitive(dummyOp, [PaddingMode], PaddingMode)
pset.addPrimitive(dummyOp, [UpsampleMode], UpsampleMode)
pset.addPrimitive(dummyOp, [SkipMergeType], SkipMergeType)
pset.addPrimitive(dummyOp, [Weights], Weights)
pset.addPrimitive(dummyOp, [BoolWeight], BoolWeight)
pset.addPrimitive(dummyOp, [ConvNeXtSize], ConvNeXtSize)
pset.addPrimitive(dummyOp, [DenseNetSize], DenseNetSize)
pset.addPrimitive(dummyOp, [EfficientNet_V2Size], EfficientNet_V2Size)
pset.addPrimitive(dummyOp, [MobileNet_V3Size], MobileNet_V3Size)
pset.addPrimitive(dummyOp, [RegNetSize], RegNetSize)
pset.addPrimitive(dummyOp, [ResNeXtSize], ResNeXtSize)
pset.addPrimitive(dummyOp, [ResNetSize], ResNetSize)
pset.addPrimitive(dummyOp, [ShuffleNet_V2Size], ShuffleNet_V2Size)
pset.addPrimitive(dummyOp, [Swin_V2Size], Swin_V2Size)
pset.addPrimitive(dummyOp, [ViTSize], ViTSize)
pset.addPrimitive(dummyOp, [Wide_ResNetSize], Wide_ResNetSize)
pset.addPrimitive(dummyOp, [bool], bool)
pset.addPrimitive(dummyOp, [AnnealStrategy], AnnealStrategy)
pset.addPrimitive(dummyOp, [CyclicLRMode], CyclicLRMode)
pset.addPrimitive(dummyOp, [CyclicLRScaleMode], CyclicLRScaleMode)

# adding ephemeral constants
pset.addEphemeralConstant("randProbFloat", partial(random.uniform, 0, 1), ProbFloat)
pset.addEphemeralConstant("randFloat", partial(random.uniform, 0, 100), float)
pset.addEphemeralConstant("randInt", partial(random.randint, 0, 100), GenericInt)
pset.addEphemeralConstant("randBool", genRandBool, bool)
pset.addEphemeralConstant("randChannel", partial(random.randint, 1, MAX_CHANNEL_SIZE), ChannelSize)
pset.addEphemeralConstant("randKernel", partial(random.randint, 1, MAX_KERNEL_SIZE), KernelSize)
pset.addEphemeralConstant("randStride", partial(random.randint, 1, MAX_STRIDE_SIZE), StrideSize)
pset.addEphemeralConstant("randPadding", partial(random.randint, 0, MAX_PADDING_SIZE), PaddingSize)
pset.addEphemeralConstant("randOutput", partial(random.randint, 1, MAX_OUTPUT_SIZE), OutputSize)
pset.addEphemeralConstant("randDilation", partial(random.randint, 1, MAX_DILATION_SIZE), DilationSize)
pset.addEphemeralConstant("randGroup", partial(random.randint, 1, MAX_GROUP_SIZE), GroupSize)
pset.addEphemeralConstant("randSkipSize", partial(random.randint, 1, MAX_SKIP_SIZE), SkipSize)
pset.addEphemeralConstant("randPNorm", partial(random.uniform, 1, MAX_PNORM_SIZE), PNorm)
pset.addEphemeralConstant("randPaddingMode", partial(random.randint, 0, len(PaddingMode)-1), PaddingMode)
pset.addEphemeralConstant("randUpsampleMode", partial(random.randint, 0, len(UpsampleMode)-1), UpsampleMode)
pset.addEphemeralConstant("randSkipMergeType", partial(random.randint, 0, len(SkipMergeType)-1), SkipMergeType)
pset.addEphemeralConstant("randWeights", partial(random.randint, 0, len(Weights)-1), Weights)
pset.addEphemeralConstant("randBoolWeight", partial(random.randint, 0, len(BoolWeight)-1), BoolWeight)
pset.addEphemeralConstant("randConvNeXtSize", partial(random.randint, 0, len(ConvNeXtSize)-1), ConvNeXtSize)
pset.addEphemeralConstant("randDenseNetSize", partial(random.randint, 0, len(DenseNetSize)-1), DenseNetSize)
pset.addEphemeralConstant("randEfficientNet_V2Size", partial(random.randint, 0, len(EfficientNet_V2Size)-1), EfficientNet_V2Size)
pset.addEphemeralConstant("randMobileNet_V3Size", partial(random.randint, 0, len(MobileNet_V3Size)-1), MobileNet_V3Size)
pset.addEphemeralConstant("randRegNetSize", partial(random.randint, 0, len(RegNetSize)-1), RegNetSize)
pset.addEphemeralConstant("randResNeXtSize", partial(random.randint, 0, len(ResNeXtSize)-1), ResNeXtSize)
pset.addEphemeralConstant("randResNetSize", partial(random.randint, 0, len(ResNetSize)-1), ResNetSize)
pset.addEphemeralConstant("randShuffleNet_V2Size", partial(random.randint, 0, len(ShuffleNet_V2Size)-1), ShuffleNet_V2Size)
pset.addEphemeralConstant("randSwin_V2Size", partial(random.randint, 0, len(Swin_V2Size)-1), Swin_V2Size)
pset.addEphemeralConstant("randViTSize", partial(random.randint, 0, len(ViTSize)-1), ViTSize)
pset.addEphemeralConstant("randWide_ResNetSize", partial(random.randint, 0, len(Wide_ResNetSize)-1), Wide_ResNetSize)
pset.addEphemeralConstant("randAnnealStrategy", partial(random.randint, 0, len(AnnealStrategy)-1), AnnealStrategy)
pset.addEphemeralConstant("randCyclicLRMode", partial(random.randint, 0, len(CyclicLRMode)-1), CyclicLRMode)
pset.addEphemeralConstant("randCyclicLRScaleMode", partial(random.randint, 0, len(CyclicLRScaleMode)-1), CyclicLRScaleMode)

pset.addTerminal(Optimizer({'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0, 'dampening': 0}), Optimizer) # default terminal for optimizer (SGD with default params)
pset.addTerminal(Scheduler({'lr_scheduler': 'StepLR', 'step_size': 30, 'gamma': 0.1}), Scheduler) # default terminal for scheduler (StepLR with default params)
