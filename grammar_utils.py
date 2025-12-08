import torch
import torch.nn.functional as F
import inspect
import enum
import primitives 
from deap import gp

# --- Bounds Mapping based on primitives.py constants ---
# We map the Class Type to the (Max Value, Is Integer) tuple.
# ProbFloat is implicit 0-1.
TYPE_BOUNDS = {
    primitives.ChannelSize: (primitives.MAX_CHANNEL_SIZE, True), # 64
    primitives.KernelSize: (primitives.MAX_KERNEL_SIZE, True),   # 9
    primitives.StrideSize: (primitives.MAX_STRIDE_SIZE, True),   # 9
    primitives.PaddingSize: (primitives.MAX_PADDING_SIZE, True), # 9
    primitives.OutputSize: (primitives.MAX_OUTPUT_SIZE, True),   # 3000
    primitives.DilationSize: (primitives.MAX_DILATION_SIZE, True), # 10
    primitives.GroupSize: (primitives.MAX_GROUP_SIZE, True),     # 1
    primitives.SkipSize: (primitives.MAX_SKIP_SIZE, True),       # 5
    primitives.GenericInt: (100, True),  # Based on randInt ephemeral
    int: (100, True),                    # Fallback for plain ints
    bool: (1, True),                     # 0 or 1
    
    # Floats
    primitives.PNorm: (primitives.MAX_PNORM_SIZE, False),        # 3.0
    primitives.BoundedFloat: (primitives.MAX_FLOAT_SIZE, False), # 100.0
    float: (primitives.MAX_FLOAT_SIZE, False)                    # Fallback
}

def get_layer_type_map(pset):
    """Returns ordered list of primitive names."""
    filtered_prims = []
    for primitive, func in pset.mapping.items():
        if str(primitive)[:2] != 'to' and str(primitive) not in ['IN0', 'add', 'mul', 'dummyOp', 'protectedDiv', 'protectedSub'] and type(func) not in [gp.Terminal, type]:
            filtered_prims.append(primitive)
    return list(filtered_prims)

def parse_primitive_schemas(pset):
    """
    Analyzes primitives to build detailed bounds schemas.
    """
    layer_names = get_layer_type_map(pset)
    schema_dict = {}
    
    for layer_idx, prim_name in enumerate(layer_names):
        params = []
        current_idx = 0
        
        # Get the actual function from primitives module by name
        sig = eval(f'inspect.signature(primitives.{prim_name})')
        
        for param_name, param_obj in sig.parameters.items():
            if param_name == 'tensor': continue 
            
            annotation = param_obj.annotation
            
            # 1. Handle Enums
            if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
                enum_len = len(annotation)
                params.append({
                    'type': 'enum',
                    'start_idx': current_idx,
                    'end_idx': current_idx + enum_len,
                    'max_val': 1.0, # Probabilities sum to 1
                    'name': param_name
                })
                current_idx += enum_len
                
            # 2. Handle ProbFloat (0.0 - 1.0)
            elif annotation is primitives.ProbFloat:
                params.append({
                    'type': 'float',
                    'start_idx': current_idx,
                    'end_idx': current_idx + 1,
                    'max_val': 1.0,
                    'name': param_name
                })
                current_idx += 1
                
            # 3. Handle Specific Bounded Types (Ints and other Floats)
            elif annotation in TYPE_BOUNDS:
                max_val, is_int = TYPE_BOUNDS[annotation]
                type_str = 'int' if is_int else 'float'
                
                params.append({
                    'type': type_str,
                    'start_idx': current_idx,
                    'end_idx': current_idx + 1,
                    'max_val': float(max_val),
                    'name': param_name
                })
                current_idx += 1
                
            # 4. Fallback (Optimizer/Scheduler structural args)
            else:
                 pass
                 
        schema_dict[layer_idx] = params
        
    return schema_dict

# Initialize Cache
PRIM_SCHEMA = parse_primitive_schemas(primitives.pset)