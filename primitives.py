from enum import Enum
from functools import partial
import operator
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
MAX_FLOAT_SIZE = 50


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

class OuputSize(int):
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

def FractionalMaxPool2d(tensor: Tensor3D, kernel_size0: KernelSize, kernel_size1: KernelSize, output_size0: OuputSize, output_size1: OuputSize):
    return Tensor3D()

def LPPool2d(tensor: Tensor3D, kernel_size0 : KernelSize, kernel_size1: KernelSize, stride0: StrideSize, stride1: StrideSize):
    return Tensor3D()

def AdaptiveMaxPool2d(tensor: Tensor3D, output_size0: OuputSize, output_size1: OuputSize):
    return Tensor3D()

def AdaptiveAvgPool2d(tensor: Tensor3D, output_size0: OuputSize, output_size1: OuputSize):
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

def Dropout_2D(tensor: Tensor3D, p: float):
    return Tensor3D()

def Dropout_1D(tensor: Tensor1D, p: float):
    return Tensor1D()


# Flatten layers
def Flatten(tensor: Tensor3D | Tensor2D | Tensor1D):
    return Tensor1D()


# Linear layers
def LazyLinear(tensor: Tensor1D, out_features: OuputSize):
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
def Detection_Head(tensor: Tensor1D):
    return FinalTensor()

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
                  [Tensor3D, KernelSize, KernelSize, OuputSize, OuputSize],
                  Tensor3D)

pset.addPrimitive(LPPool2d,
                  [Tensor3D, KernelSize, KernelSize, StrideSize, StrideSize],
                  Tensor3D)

pset.addPrimitive(AdaptiveMaxPool2d,
                  [Tensor3D, OuputSize, OuputSize],
                  Tensor3D)

pset.addPrimitive(AdaptiveAvgPool2d,
                  [Tensor3D, OuputSize, OuputSize],
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

pset.addPrimitive(RReLU_2D,
                  [Tensor3D, float, float],
                  Tensor3D)

pset.addPrimitive(RReLU_1D,
                  [Tensor1D, float, float],
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
                  [Tensor3D, float],
                  Tensor3D)

pset.addPrimitive(Dropout_1D,
                  [Tensor1D, float],
                  Tensor1D)

pset.addPrimitive(Upsample_1D,
                  [Tensor1D, float, UpsampleMode],
                  Tensor1D)

pset.addPrimitive(Skip_2D,
                  [Tensor3D, SkipSize, SkipMergeType],
                  Tensor3D)

pset.addPrimitive(Detection_Head,
                  [Tensor1D],
                  FinalTensor)

pset.addPrimitive(Flatten,
                  [Tensor3D],
                  Tensor1D)

pset.addPrimitive(LazyLinear,
                  [Tensor1D, OuputSize],
                  Tensor1D)

pset.addPrimitive(Upsample_2D,
                  [Tensor3D, float, UpsampleMode],
                  Tensor3D)

pset.addPrimitive(Skip_1D,
                  [Tensor1D, SkipSize, SkipMergeType],
                  Tensor1D)

def channelAdd(a, b):
    return ChannelSize((a+b)%MAX_CHANNEL_SIZE)

def kernelAdd(a, b):
    return KernelSize((a+b)%MAX_KERNEL_SIZE)

def strideAdd(a, b):
    return StrideSize((a+b)%MAX_STRIDE_SIZE)

def paddingAdd(a, b):
    return PaddingSize((a+b)%MAX_PADDING_SIZE)

def outputAdd(a, b):
    return OuputSize((a+b)%MAX_OUTPUT_SIZE)

def dilationAdd(a, b):
    return DilationSize((a+b)%MAX_DILATION_SIZE)

def groupAdd(a, b):
    return GroupSize((a+b)%MAX_GROUP_SIZE)

def skipAdd(a, b):
    return SkipSize((a+b)%MAX_SKIP_SIZE)

def floatAdd(a, b):
    return (a+b)%MAX_FLOAT_SIZE

pset.addPrimitive(channelAdd, [ChannelSize, ChannelSize], ChannelSize)
pset.addPrimitive(kernelAdd, [KernelSize, KernelSize], KernelSize)
pset.addPrimitive(strideAdd, [StrideSize, StrideSize], StrideSize)
pset.addPrimitive(paddingAdd, [PaddingSize, PaddingSize], PaddingSize)
pset.addPrimitive(outputAdd, [OuputSize, OuputSize], OuputSize)
pset.addPrimitive(dilationAdd, [DilationSize, DilationSize], DilationSize)
pset.addPrimitive(groupAdd, [GroupSize, GroupSize], GroupSize)
pset.addPrimitive(skipAdd, [SkipSize, SkipSize], SkipSize)
pset.addPrimitive(floatAdd, [float, float], float)

def protectedSub(a, b):
    return abs(a-b)

pset.addPrimitive(protectedSub, [ChannelSize, ChannelSize], ChannelSize)
pset.addPrimitive(protectedSub, [KernelSize, KernelSize], KernelSize)
pset.addPrimitive(protectedSub, [StrideSize, StrideSize], StrideSize)
pset.addPrimitive(protectedSub, [PaddingSize, PaddingSize], PaddingSize)
pset.addPrimitive(protectedSub, [OuputSize, OuputSize], OuputSize)
pset.addPrimitive(protectedSub, [DilationSize, DilationSize], DilationSize)
pset.addPrimitive(protectedSub, [GroupSize, GroupSize], GroupSize)
pset.addPrimitive(protectedSub, [SkipSize, SkipSize], SkipSize)
pset.addPrimitive(protectedSub, [float, float], float)

def channelMul(a, b):
    return ChannelSize((a*b)%MAX_CHANNEL_SIZE)

def kernelMul(a, b):
    return KernelSize((a*b)%MAX_KERNEL_SIZE)

def strideMul(a, b):
    return StrideSize((a*b)%MAX_STRIDE_SIZE)

def paddingMul(a, b):
    return PaddingSize((a*b)%MAX_PADDING_SIZE)

def outputMul(a, b):
    return OuputSize((a*b)%MAX_OUTPUT_SIZE)

def dilationMul(a, b):
    return DilationSize((a*b)%MAX_DILATION_SIZE)

def groupMul(a, b):
    return GroupSize((a*b)%MAX_GROUP_SIZE)

def skipMul(a, b):
    return SkipSize((a*b)%MAX_SKIP_SIZE)

def floatMul(a, b):
    return (a*b)%MAX_FLOAT_SIZE

pset.addPrimitive(channelMul, [ChannelSize, ChannelSize], ChannelSize)
pset.addPrimitive(kernelMul, [KernelSize, KernelSize], KernelSize)
pset.addPrimitive(strideMul, [StrideSize, StrideSize], StrideSize)
pset.addPrimitive(paddingMul, [PaddingSize, PaddingSize], PaddingSize)
pset.addPrimitive(outputMul, [OuputSize, OuputSize], OuputSize)
pset.addPrimitive(dilationMul, [DilationSize, DilationSize], DilationSize)
pset.addPrimitive(groupMul, [GroupSize, GroupSize], GroupSize)
pset.addPrimitive(skipMul, [SkipSize, SkipSize], SkipSize)
pset.addPrimitive(floatMul, [float, float], float)

def protectedDiv(left, right):
    if (isinstance(left, int) and isinstance(right, int)):
        try: return left // right
        except ZeroDivisionError: return 1
    else:
        try: return left / right
        except ZeroDivisionError: return 1

pset.addPrimitive(protectedDiv, [ChannelSize, ChannelSize], ChannelSize)
pset.addPrimitive(protectedDiv, [KernelSize, KernelSize], KernelSize)
pset.addPrimitive(protectedDiv, [StrideSize, StrideSize], StrideSize)
pset.addPrimitive(protectedDiv, [PaddingSize, PaddingSize], PaddingSize)
pset.addPrimitive(protectedDiv, [OuputSize, OuputSize], OuputSize)
pset.addPrimitive(protectedDiv, [DilationSize, DilationSize], DilationSize)
pset.addPrimitive(protectedDiv, [GroupSize, GroupSize], GroupSize)
pset.addPrimitive(protectedDiv, [SkipSize, SkipSize], SkipSize)
pset.addPrimitive(protectedDiv, [float, float], float)

def dummyOp(input):
    return input

pset.addPrimitive(dummyOp, [PaddingMode], PaddingMode)
pset.addPrimitive(dummyOp, [UpsampleMode], UpsampleMode)
pset.addPrimitive(dummyOp, [SkipMergeType], SkipMergeType)

pset.addEphemeralConstant("randChannel", partial(random.randint, 1, MAX_CHANNEL_SIZE), ChannelSize)
pset.addEphemeralConstant("randKernel", partial(random.randint, 1, MAX_KERNEL_SIZE), KernelSize)
pset.addEphemeralConstant("randStride", partial(random.randint, 1, MAX_STRIDE_SIZE), StrideSize)
pset.addEphemeralConstant("randPadding", partial(random.randint, 1, MAX_PADDING_SIZE), PaddingSize)
pset.addEphemeralConstant("randOutput", partial(random.randint, 1, MAX_OUTPUT_SIZE), OuputSize)
pset.addEphemeralConstant("randDilation", partial(random.randint, 1, MAX_DILATION_SIZE), DilationSize)
pset.addEphemeralConstant("randGroup", partial(random.randint, 1, MAX_GROUP_SIZE), GroupSize)
pset.addEphemeralConstant("randSkipSize", partial(random.randint, 1, MAX_SKIP_SIZE), SkipSize)
pset.addEphemeralConstant("randFloat", partial(random.uniform, 0, 1), float) # note that there might be some places where floats outside this range are valid.
pset.addEphemeralConstant("randPaddingMode", partial(random.randint, 0, len(PaddingMode)-1), PaddingMode)
pset.addEphemeralConstant("randUpsampleMode", partial(random.randint, 0, len(UpsampleMode)-1), UpsampleMode)
pset.addEphemeralConstant("randSkipMergeType", partial(random.randint, 0, len(SkipMergeType)-1), SkipMergeType)

pset.addTerminal(Tensor1D(), Tensor1D) # seeing a terminal requires us to add a special cell (replace Tensor1D() with custom cell definition later)
