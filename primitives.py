from enum import Enum

from deap import gp


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

def LazyConvTransposed2d(tensor: Tensor3D, out_channels: ChannelSize, kernel_size0: KernelSize, kernel_size1: KernelSize, stride0: StrideSize, stride1: StrideSize, padding0: PaddingSize, padding1: PaddingSize, padding_mode: PaddingMode, 
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

def Softmax(tensor: Tensor1D, dim: OuputSize):
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
def Skip_2D(tensor: Tensor3D, skip_by: int, merge_type: SkipMergeType):
    return Tensor3D()

def Skip_1D(tensor: Tensor1D, skip_by: int, merge_type: SkipMergeType):
    return Tensor1D()

# TODO: cells
def Detection_Head(tensor: Tensor1D):
    return FinalTensor()

# creating primitive set
pset = gp.PrimitiveSetTyped("MAIN", [Tensor3D], FinalTensor)
pset.addPrimitive(LazyConv2d, 
                  [Tensor3D, ChannelSize, KernelSize, KernelSize, StrideSize, StrideSize, PaddingSize, PaddingSize, PaddingMode, DilationSize, DilationSize, GroupSize], 
                  Tensor3D)

pset.addPrimitive(LazyConvTransposed2d,
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
                  [Tensor1D, OuputSize],
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

# breakpoint() investigate ordering issue

pset.addPrimitive(Upsample_1D,
                  [Tensor1D, float, UpsampleMode],
                  Tensor1D)

pset.addPrimitive(Skip_2D,
                  [Tensor3D, int, SkipMergeType],
                  Tensor3D)

pset.addPrimitive(Detection_Head,
                  [Tensor1D],
                  FinalTensor)

pset.addPrimitive(Flatten,
                  [Tensor3D | Tensor2D | Tensor1D],
                  Tensor1D)

pset.addPrimitive(LazyLinear,
                  [Tensor1D, OuputSize],
                  Tensor1D)

pset.addPrimitive(Upsample_2D,
                  [Tensor3D, float, UpsampleMode],
                  Tensor3D)

pset.addPrimitive(Skip_1D,
                  [Tensor1D, int, SkipMergeType],
                  Tensor1D)

# TODO: need to add more math operator primitives and ephemeral constants 