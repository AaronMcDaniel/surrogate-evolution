"""
Individual Inspector - Converts genome strings to structured reports for LLM analysis.

This module provides functionality to parse genome strings and generate detailed
reports about the neural architecture structure, including layer configurations,
hyperparameters, and component hierarchies.
"""

from deap import creator, gp, base, tools
from primitive_tree import CustomPrimitiveTree
import primitives



def ensure_deap_classes():
    """Initialize DEAP creator classes if they don't exist."""
    if not hasattr(creator, 'FitnessMulti'):
        creator.create("FitnessMulti", base.Fitness, weights=(-1, -1, 1))
        genome_type = gp.PrimitiveTree

    if not hasattr(creator, 'Individual'):
        creator.create("Individual", genome_type, fitness=creator.FitnessMulti)

def inspect_individual_to_string(genome_string: str) -> str:
    global primitives_code, codec_code
    """
    Convert a genome string into a structured report for LLM interpretation.
    
    This function parses a genome string representation of a neural architecture
    and generates a detailed, hierarchical report of all components, their types,
    arities, return types, and arguments.
    
    Args:
        genome_string (str): A string representation of a neural architecture genome,
                           e.g., "RetinaNet_Head(ConvNeXt(...))"
    
    Returns:
        str: A formatted report string with tabular structure showing:
             - Index position of each node
             - Node class (Primitive or Terminal)
             - Node name (operation/layer name)
             - Arity (number of inputs)
             - Return type
             - Arguments or values
    
    Example:
        >>> genome = "RetinaNet_Head(ConvNeXt(AdaptiveAvgPool2d(...)))"
        >>> report = inspect_individual_to_string(genome)
        >>> print(report)
    """
    # Ensure DEAP classes are initialized
    ensure_deap_classes()
    
    # Initialize toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitives.pset, min_=4, max_=8)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Parse the genome string into an individual
    individual = creator.Individual(CustomPrimitiveTree.from_string(genome_string, primitives.pset))
    
    # Build the report string
    report_lines = []
    
    # Add header
    report_lines.append("=" * 100)
    report_lines.append("NEURAL ARCHITECTURE GENOME INSPECTION REPORT")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"Genome String: {genome_string}")
    report_lines.append("")
    report_lines.append("Node-by-Node Breakdown:")
    report_lines.append("-" * 100)
    
    # Table header
    header = f"{'IDX':<4} | {'CLASS':<10} | {'NAME':<20} | {'ARITY':<5} | {'RET TYPE':<15} | {'DETAILS'}"
    report_lines.append(header)
    report_lines.append("-" * 100)
    
    # Process each node
    for i, node in enumerate(individual):
        # Identify node class
        node_class = "Primitive" if isinstance(node, gp.Primitive) else "Terminal"
        
        # Extract return type
        ret_type = getattr(node.ret, '__name__', str(node.ret))
        
        # Extract node-specific details
        if isinstance(node, gp.Primitive):
            # Primitives have argument types
            arg_types = [getattr(arg, '__name__', str(arg)) for arg in node.args]
            details = f"Inputs: [{', '.join(arg_types)}]"
        else:
            # Terminals have values
            details = f"Value: {node.value}"
        
        # Format and add row
        row = f"{i:<4} | {node_class:<10} | {node.name:<20} | {node.arity:<5} | {ret_type:<15} | {details}"
        report_lines.append(row)
    
    # Add summary section
    report_lines.append("-" * 100)
    report_lines.append("")
    report_lines.append("SUMMARY:")
    report_lines.append(f"  Total Nodes: {len(individual)}")
    
    # Count primitives and terminals
    num_primitives = sum(1 for node in individual if isinstance(node, gp.Primitive))
    num_terminals = len(individual) - num_primitives
    report_lines.append(f"  Primitives (Operations): {num_primitives}")
    report_lines.append(f"  Terminals (Leaf Nodes): {num_terminals}")
    
    # Identify unique operations
    unique_ops = set(node.name for node in individual if isinstance(node, gp.Primitive))
    report_lines.append(f"  Unique Operations: {len(unique_ops)}")
    report_lines.append(f"  Operation Types: {', '.join(sorted(unique_ops))}")
    
    report_lines.append("")

    report_lines.append("-" * 100)
    report_lines.append("ARCHITECTURE SEARCH SPACE CONTEXT")
    report_lines.append("-" * 100)
    report_lines.append("")
    report_lines.append("The following sections provide context about the architecture search space,")
    report_lines.append("including primitive definitions and genome decoding logic powered by")
    report_lines.append("the 'Distributed Evolutionary Algorithms in Python' library.")
    report_lines.append("")
    report_lines.append("--- PRIMITIVES DEFINITIONS ---")
    report_lines.append("```python")
    report_lines.append(primitives_code)
    report_lines.append("```")
    report_lines.append("")
    report_lines.append("--- GENOME STRING TO PYTORCH MODEL OBJECT CONVERSION LOGIC ---")
    report_lines.append("```python")
    report_lines.append(codec_code)
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100)
    
    return "\n".join(report_lines)

codec_code = r"""def decode_genome(self, genome, num_loss_components):
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
                        stack.append(str(eval(f'primitives.{function}({",".join(arguments)})')))
                    except: # this is where we add the layers
                        layer_info = [function]+[self.__parse_arg(x) for x in arguments]
                        info = self.add_to_module_list(module_list, idx, layer_info, num_loss_components)
                        idx += 1

            head = info[0]
            model_dict = info[1]
            skip_info = torch.randn(4,4)
            model = self.add_head(head, module_list, skip_info)
            model_dict['model'] = model
            return model_dict
        

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
            module_list.append(eval(f'nn.{layer_name.split("_")[0]}')(
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
            module_list.append(eval(f'nn.{layer_name.split("_")[0]}')(
                    output_size=(layer_args[0], layer_args[1]),
                ))
        
        elif layer_name in ['Upsample_1D', 'Upsample_2D']:
            module_list.append(eval(f'nn.{layer_name.split("_")[0]}')(
                    scale_factor=layer_args[0],
                    mode=(list(primitives.UpsampleMode)[layer_args[1]]).name
                ))
            
        elif layer_name in ['Skip_1D', 'Skip_2D']:
            pass # TODO implement skip layer

        # detection head layer
        elif layer_name in HEADS:
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
            return (layer_name, out_dict)
            
        else: # this is for layers that can have arguments simply unpacked
            module_list.append(eval(f'nn.{layer_name.split("_")[0]}')(*layer_args))
    

    # function to add appropriate detection head to custom backbone
    def add_head(self, head, module_list, skip_info):
        dummy_input = torch.randn(1, 3, 2048, 2448).to(self.device)
        model = DynamicNetwork(module_list, skip_info)
        test_model = model.to(self.device)
        output = test_model(dummy_input)
        model.out_channels = output.shape[1]
        if head == 'FasterRCNN_Head':
            anchor_generator = RCNNAnchorGenerator(
                sizes=((4, 8, 16, 32, 64, 128, 256),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
            roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                featmap_names=['0'],
                output_size=21,
                sampling_ratio=4
            )
            model = CustomFasterRCNN(
                model,
                num_classes=self.num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
                box_score_thresh=0,
                box_nms_thresh=1, 
                min_size=1200,
                max_size=2000,
                box_detections_per_img=100
            )
        if head == 'FCOS_Head':
            anchor_generator = AnchorGenerator(
                sizes=((4,), (8,), (16,), (32,), (64,), (128,), (256,), (512,)),
                aspect_ratios=((1.0,),)
            )
            model = CustomFCOS(
                model,
                num_classes=self.num_classes,
                anchor_generator=anchor_generator,
                score_thresh=0,
                nms_thresh=1,
                min_size=1200,
                max_size=2000,
                detections_per_img=100
            )
        if head == 'RetinaNet_Head':
            anchor_generator = AnchorGenerator(
                sizes=((4, 8, 16, 32, 64, 128, 256),),
                aspect_ratios=((0.5, 1.0, 2.0),)
            )
            model = CustomRetinaNet(
                model,
                num_classes=self.num_classes,
                anchor_generator=anchor_generator,
                score_thresh=0,
                nms_thresh=1,
                min_size=1200,
                max_size=2000,
                detections_per_img=100
            )
        if head == 'SSD_Head':
            anchor_generator = DefaultBoxGenerator(
                aspect_ratios=[(0.5, 1.0, 2.0)],
                scales=[8, 16, 32, 64, 128, 256]
            )
            model = SSD(
                model,
                num_classes=self.num_classes,
                anchor_generator=anchor_generator
            )   
        return model   
"""

primitives_code = r"""model_config = configs["model"]
num_loss_components = int(model_config['num_loss_components'])


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
MAX_FLOAT_SIZE = 100


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

class BoundedFloat(float):
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

def LeakyReLU_2D(tensor: Tensor3D, negative_slope: BoundedFloat):
    return Tensor3D()

def RReLU_2D(tensor: Tensor3D, lower: BoundedFloat, upper: BoundedFloat):
    return Tensor3D()

def LogSigmoid_2D(tensor: Tensor3D):
    return Tensor3D()

def Sigmoid_2D(tensor: Tensor3D):
    return Tensor3D()

def Tanh_2D(tensor: Tensor3D):
    return Tensor3D()

def Threshold_2D(tensor: Tensor3D, threshold: BoundedFloat, value: BoundedFloat):
    return Tensor3D()


# TODO: MultiHeadAttention


# Normalization and Dropout layers
def LazyBatchNorm2d(tensor: Tensor3D, eps: BoundedFloat, momentum: BoundedFloat):
    return Tensor3D()

def Dropout_2D(tensor: Tensor3D, p: ProbFloat):
    return Tensor3D()


# TODO: transformer layers


# Vision layers
def Upsample_2D(tensor: Tensor3D, scaling_factor: BoundedFloat, mode: UpsampleMode):
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
    eta_lower = transform_value(eta_lower, 0.5, 1)
    eta_upper = transform_value(eta_upper, 1.1, 1.5)
    step_lower = transform_value(step_lower, 1e-6, 1)
    step_upper = transform_value(step_upper, 1, 50)
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
                  [Tensor3D, BoundedFloat],
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
                  [Tensor3D, BoundedFloat, BoundedFloat],
                  Tensor3D)

pset.addPrimitive(LazyBatchNorm2d,
                  [Tensor3D, BoundedFloat, BoundedFloat],
                  Tensor3D)

pset.addPrimitive(Dropout_2D,
                  [Tensor3D, ProbFloat],
                  Tensor3D)

# pset.addPrimitive(Skip_2D,
#                   [Tensor3D, SkipSize, SkipMergeType],
#                   Tensor3D)

pset.addPrimitive(FasterRCNN_Head,
                  [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(ProbFloat, num_loss_components)),
                  FinalTensor)

pset.addPrimitive(FCOS_Head,
                  [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(ProbFloat, num_loss_components)),
                  FinalTensor)

pset.addPrimitive(RetinaNet_Head,
                  [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(ProbFloat, num_loss_components)),
                  FinalTensor)

# pset.addPrimitive(SSD_Head,
#                   [Tensor3D, Optimizer, Scheduler] + list(itertools.repeat(float, num_loss_components)),
#                   FinalTensor)

pset.addPrimitive(Upsample_2D,
                  [Tensor3D, BoundedFloat, UpsampleMode],
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

def toBoundedFloat(a):
    return a%MAX_FLOAT_SIZE

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
"""



if __name__ == "__main__":
    # Test with example genome
    test_genome = "RetinaNet_Head(ConvNeXt(AdaptiveAvgPool2d(LeakyReLU_2D(IN0, 96.04792006022558), 1547, 964), 0, 1), SGD(3.604584617149115, 86.64620664538738, 97.79196427135827, 0.1077832694243166), MultiStepLR(4.863611087893961), 0.9894983028181139, 0.007501546652931346, 0.10196727020619534, 0.5854467195745006, 0.01466235249701516, 0.07275973157192794, 0.7396087325577554)"
    
    print("DETAILED REPORT:")
    print(inspect_individual_to_string(test_genome))
