from enum import Enum
from functools import partial
import random

from deap import gp


MAX_CHANNEL_SIZE = 64
MAX_KERNEL_SIZE = 9
MAX_STRIDE_SIZE = 9
MAX_PADDING_SIZE = 9
MAX_OUTPUT_SIZE = 3000
MAX_DILATION_SIZE = 10
MAX_GROUP_SIZE = 1
MAX_SKIP_SIZE = 5
MAX_PNORM_SIZE = 3

'''
NOTE: 1D and 2D TENSOR LAYERS ARE NOW OBSOLETE AND SHOULD BE REMOVED AT SOME POINT
'''

# placeholder classes to act as types for DEAP's strongly typed primitive set
class Tensor3D:
    pass

class Tensor2D:
    pass

class Tensor1D:
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

class GenericInt(int):
    def __init__(self, num) -> None:
        super().__init__()
    pass

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

class ConvNextSize(Enum):
    Base = 0,
    Large = 1,
    Small = 2,
    Tiny = 3

class DenseNetSize(Enum):
    _121 = 0,
    _161 = 1,
    _169 = 2,
    _201 = 3

class EfficientNetSize(Enum):
    B0 = 0,
    B1 = 1,
    B2 = 2,
    B3 = 3,
    B4 = 4,
    B5 = 5,
    B6 = 6,
    B7 = 7

class MobileNetSize(Enum):
    Large = 0,
    Small = 1

class RegNetSize(Enum):
    pass


class Optimizer(Enum):
    Adelta = 0,
    Adagrad = 1,
    Adam = 2,
    AdamW = 3,
    SparseAdam = 4,
    Adamax = 5,
    ASGD = 6,
    LBFGS = 7,
    NAdam = 8,
    RAdam = 9,
    RMSprop = 10,
    Rprop = 11,
    SGD = 12


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

def ReLU_1D(tensor: Tensor1D):
    return Tensor1D()

def LeakyReLU_2D(tensor: Tensor3D, negative_slope: float):
    return Tensor3D()

def LeakyReLU_1D(tensor: Tensor1D, negative_slope: float):
    return Tensor1D()

def RReLU_2D(tensor: Tensor3D, lower: float, upper: float):
    return Tensor3D()

def RReLU_1D(tensor: Tensor1D, lower: float, upper: float):
    return Tensor1D()

def LogSigmoid_2D(tensor: Tensor3D):
    return Tensor3D()

def LogSigmoid_1D(tensor: Tensor1D):
    return Tensor1D()

def Sigmoid_2D(tensor: Tensor3D):
    return Tensor3D()

def Sigmoid_1D(tensor: Tensor1D):
    return Tensor1D()

def Tanh_2D(tensor: Tensor3D):
    return Tensor3D()

def Tanh_1D(tensor: Tensor1D):
    return Tensor1D()

def Threshold_2D(tensor: Tensor3D, threshold: float, value: float):
    return Tensor3D()

def Threshold_1D(tensor: Tensor1D, threshold: float, value: float):
    return Tensor1D()

def Softmax(tensor: Tensor1D):
    return Tensor1D()


# TODO: MultiHeadAttention


# Normalization and Dropout layers
def LazyBatchNorm2d(tensor: Tensor3D, eps: float, momentum: float):
    return Tensor3D()

def Dropout_2D(tensor: Tensor3D, p: ProbFloat):
    return Tensor3D()

def Dropout_1D(tensor: Tensor1D, p: ProbFloat):
    return Tensor1D()


# Flatten layers
def Flatten(tensor: Tensor3D | Tensor2D | Tensor1D):
    return Tensor1D()


# Linear layers
def LazyLinear(tensor: Tensor1D, out_features: OutputSize):
    return Tensor1D()


# TODO: transformer layers


# Vision layers
def Upsample_1D(tensor: Tensor1D, scaling_factor: float, mode: UpsampleMode):
    return Tensor1D()
    
def Upsample_2D(tensor: Tensor3D, scaling_factor: float, mode: UpsampleMode):
    return Tensor3D()


# Skip Connection Support: The way this is supposed to work is that a Skip layer can be added anywhere and the skip_by parameter tells us how many layers to skip
# by which allows us to then use the chosen merge_type to merge after skipping. It is likely that the merging will not be straighforward since dimensions may vary
# and the merge location may not even exist if a bad skip_by value is chosen, but we can either heal when decoding or hope the GA will figure out how to use padding 
# layers or similar to make it work.  
def Skip_2D(tensor: Tensor3D, skip_by: SkipSize, merge_type: SkipMergeType):
    return Tensor3D()

def Skip_1D(tensor: Tensor1D, skip_by: SkipSize, merge_type: SkipMergeType):
    return Tensor1D()

# TODO: cells
def Detection_Head(tensor: Tensor3D, optimizer: Optimizer, lr: float, iou_thresh: ProbFloat, conf_thresh: ProbFloat, iou_weight: float, diou_weight: float,
                   giou_weight: float, ciou_weight: float, precision_weight: float, recall_weight: float, ap_weight: float, center_l2_weight: float, 
                   area_l2_weight: float):
    return FinalTensor()

def ConvNext(tensor: Tensor3D, convnextsize: ConvNextSize):
    return Tensor3D()

def DenseNet(tensor: Tensor3D, densenetsize: DenseNetSize):
    return Tensor3D()

def EfficientNet(tensor: Tensor3D):
    return Tensor3D()

def Inception_V3(tensor: Tensor3D):
    return Tensor3D()

def MaxViT_T(tensor: Tensor3D):
    return Tensor3D()
    
def MobileNet_V3(tensor: Tensor3D):
    return Tensor3D()

def RegNet_X(tensor: Tensor3D):
    return Tensor3D()

def RegNet_Y(tensor: Tensor3D):
    return Tensor3D()

def ResNeXt(tensor: Tensor3D):
    return Tensor3D()

def ResNet(tensor: Tensor3D):
    return Tensor3D()

def ShuffleNet(tensor: Tensor3D):
    return Tensor3D()

def Swin_V2(tensor: Tensor3D):
    return Tensor3D()

def ViT(tensor: Tensor3D):
    return Tensor3D()

def Wide_ResNet(tensor: Tensor3D):
    return Tensor3D()

# creating primitive set
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

pset.addPrimitive(ReLU_1D,
                  [Tensor1D],
                  Tensor1D)

pset.addPrimitive(LeakyReLU_2D,
                  [Tensor3D, float],
                  Tensor3D)

pset.addPrimitive(LeakyReLU_1D,
                  [Tensor1D, float],
                  Tensor1D)

pset.addPrimitive(LogSigmoid_2D,
                  [Tensor3D],
                  Tensor3D)

pset.addPrimitive(LogSigmoid_1D,
                  [Tensor1D],
                  Tensor1D)

pset.addPrimitive(Sigmoid_2D,
                  [Tensor3D],
                  Tensor3D)

pset.addPrimitive(Sigmoid_1D,
                  [Tensor1D],
                  Tensor1D)

pset.addPrimitive(Tanh_2D,
                  [Tensor3D],
                  Tensor3D)

pset.addPrimitive(Tanh_1D,
                  [Tensor1D],
                  Tensor1D)

pset.addPrimitive(Threshold_2D,
                  [Tensor3D, float, float],
                  Tensor3D)

pset.addPrimitive(Threshold_1D,
                  [Tensor1D, float, float],
                  Tensor1D)

pset.addPrimitive(Softmax,
                  [Tensor1D],
                  Tensor1D)

pset.addPrimitive(LazyBatchNorm2d,
                  [Tensor3D, float, float],
                  Tensor3D)

pset.addPrimitive(Dropout_2D,
                  [Tensor3D, ProbFloat],
                  Tensor3D)

pset.addPrimitive(Dropout_1D,
                  [Tensor1D, ProbFloat],
                  Tensor1D)

pset.addPrimitive(Upsample_1D,
                  [Tensor1D, float, UpsampleMode],
                  Tensor1D)

pset.addPrimitive(Skip_2D,
                  [Tensor3D, SkipSize, SkipMergeType],
                  Tensor3D)

pset.addPrimitive(Detection_Head,
                  [Tensor3D, Optimizer, float, ProbFloat, ProbFloat, float, float, float, float, float, float, float, float, float],
                  FinalTensor)

pset.addPrimitive(Flatten,
                  [Tensor3D],
                  Tensor1D)

pset.addPrimitive(LazyLinear,
                  [Tensor1D, OutputSize],
                  Tensor1D)

pset.addPrimitive(Upsample_2D,
                  [Tensor3D, float, UpsampleMode],
                  Tensor3D)

pset.addPrimitive(Skip_1D,
                  [Tensor1D, SkipSize, SkipMergeType],
                  Tensor1D)

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

def toChannel(a):
    if a == 0:
        return 1
    return ChannelSize(a%MAX_CHANNEL_SIZE)+1 if a > MAX_CHANNEL_SIZE else ChannelSize(a)

def toKernel(a):
    if a == 0: return 3
    if a < 3: return 3
    return KernelSize(a%MAX_KERNEL_SIZE)+3 if a > MAX_KERNEL_SIZE else KernelSize(a)

def toStride(a):
    if a == 0:
        return 1
    return StrideSize(a%MAX_STRIDE_SIZE)+1 if a > MAX_STRIDE_SIZE else StrideSize(a)

def toPadding(a):
    return PaddingSize(a%MAX_PADDING_SIZE) if a > MAX_PADDING_SIZE else PaddingSize(a)

def toOutput(a):
    if a == 0:
        return 1
    return OutputSize(a%MAX_OUTPUT_SIZE)+1 if a > MAX_OUTPUT_SIZE else OutputSize(a)

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
    if a == 0:
        return 2
    return PNorm(a%MAX_PNORM_SIZE)+2 if a > MAX_PNORM_SIZE else PNorm(a)

def toProbFloat(a):
    return a%1

def dummyOp(input):
    return input

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
pset.addPrimitive(dummyOp, [Optimizer], Optimizer)

pset.addEphemeralConstant("randChannel", partial(random.randint, 1, MAX_CHANNEL_SIZE), ChannelSize)
pset.addEphemeralConstant("randKernel", partial(random.randint, 1, MAX_KERNEL_SIZE), KernelSize)
pset.addEphemeralConstant("randStride", partial(random.randint, 1, MAX_STRIDE_SIZE), StrideSize)
pset.addEphemeralConstant("randPadding", partial(random.randint, 0, MAX_PADDING_SIZE), PaddingSize)
pset.addEphemeralConstant("randOutput", partial(random.randint, 1, MAX_OUTPUT_SIZE), OutputSize)
pset.addEphemeralConstant("randDilation", partial(random.randint, 1, MAX_DILATION_SIZE), DilationSize)
pset.addEphemeralConstant("randGroup", partial(random.randint, 1, MAX_GROUP_SIZE), GroupSize)
pset.addEphemeralConstant("randSkipSize", partial(random.randint, 1, MAX_SKIP_SIZE), SkipSize)
pset.addEphemeralConstant("randPNorm", partial(random.uniform, 1, MAX_PNORM_SIZE), PNorm)
pset.addEphemeralConstant("randProbFloat", partial(random.uniform, 0, 1), ProbFloat)
pset.addEphemeralConstant("randFloat", partial(random.uniform, 0, 10), float)
pset.addEphemeralConstant("randInt", partial(random.randint, 0, 10), GenericInt)
pset.addEphemeralConstant("randPaddingMode", partial(random.randint, 0, len(PaddingMode)-1), PaddingMode)
pset.addEphemeralConstant("randUpsampleMode", partial(random.randint, 0, len(UpsampleMode)-1), UpsampleMode)
pset.addEphemeralConstant("randSkipMergeType", partial(random.randint, 0, len(SkipMergeType)-1), SkipMergeType)
pset.addEphemeralConstant("randOptimizerType", partial(random.randint, 0, len(Optimizer)-1), Optimizer)
