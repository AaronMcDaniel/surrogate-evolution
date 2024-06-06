from enum import Enum


# placeholder classes to act as types for DEAP's strongly typed primitive set
class Tensor3D:
    pass

class Tensor2D:
    pass

class Tensor1D:
    pass

class FinalTensor: # tensor going into the final output layer
    pass


# separate wrapper classes for different hyperparameters so bounds can be placed and crossovers are valid
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
def ReLU(tensor: Tensor3D | Tensor2D | Tensor1D):
    return (type(tensor))()

def ReLU6(tensor: Tensor3D | Tensor2D | Tensor1D):
    return (type(tensor))()

def LeakyReLU(tensor: Tensor3D | Tensor2D | Tensor1D, negative_slope: float):
    return (type(tensor))()

def RReLU(tensor: Tensor3D | Tensor2D | Tensor1D, lower: float, upper: float):
    return (type(tensor))()

def LogSigmoid(tensor: Tensor3D | Tensor2D | Tensor1D):
    return (type(tensor))()

def Sigmoid(tensor: Tensor3D | Tensor2D | Tensor1D):
    return (type(tensor))()

def Tanh(tensor: Tensor3D | Tensor2D | Tensor1D):
    return (type(tensor))()

def Threshold(tensor: Tensor3D | Tensor2D | Tensor1D, threshold: float, value: float):
    return (type(tensor))()

# can only be used as the final classification layer
def Softmax(tensor: Tensor1D, dim: OuputSize):
    return FinalTensor()


# TODO: MultiHeadAttention


# Normalization and Dropout layers
def LazyBatchNorm2d(tensor: Tensor3D, eps: float, momentum: float):
    return Tensor3D()

def Sigmoid(tensor: Tensor3D | Tensor2D | Tensor1D, p: float):
    return (type(tensor))()


# Flatten layers
def Flatten(tensor: Tensor3D | Tensor2D | Tensor1D):
    return Tensor1D()


# Linear layers
def LazyLinear(tensor: Tensor1D, out_features: OuputSize):
    return Tensor1D()

def LazyLinear_Final(tensor: Tensor1D, out_features: OuputSize): # final linear layer
    return FinalTensor()


# TODO: transformer layers


# Vision layers
def Upsample_1D(tensor: Tensor1D, size: OuputSize, scaling_factor: float, mode: UpsampleMode):
    return Tensor1D
    
def Upsample_2D(tensor: Tensor2D, size: OuputSize, scaling_factor: float, mode: UpsampleMode):
    return Tensor2D

def Upsample_3D(tensor: Tensor3D, size: OuputSize, scaling_factor: float, mode: UpsampleMode):
    return Tensor3D


# Skip Connection Support
# def Split(tensor: Tensor3D | Tensor2D | Tensor1D, )

# TODO: cells
