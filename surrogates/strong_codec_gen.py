from collections import OrderedDict

def generate_vector_representation(node_details: str, enum_details: str = ""):
    attribute_set = OrderedDict()
    mapping = {}
    
    # Parse enum details
    enums = {}
    for line in enum_details.strip().split("\n"):
        if "-" in line:
            enum_name, count = line.split("-")
            enums[enum_name.strip()] = int(count.strip())
    
    # Parse the input
    for line in node_details.strip().split("\n"):
        node_type, attributes = line.split(":")
        node_type = node_type.strip()
        attributes = [attr.strip() for attr in attributes.split(",") if attr.strip()]
        
        # Assign unique positions to each attribute, allowing duplicates
        if node_type not in mapping:
            mapping[node_type] = []
        
        attribute_counts = {}
        for attr in attributes:
            if attr in enums:  # Handle one-hot encoding for enums
                for i in range(enums[attr]):
                    attr_key = f"{attr}_{i}"
                    if attr_key not in attribute_set:
                        attribute_set[attr_key] = len(attribute_set)
                    mapping[node_type].append(attribute_set[attr_key])
            else:  # Handle regular attributes
                attr_key = f"{attr}_{attribute_counts.get(attr, 0)}"
                attribute_counts[attr] = attribute_counts.get(attr, 0) + 1
                
                if attr_key not in attribute_set:
                    attribute_set[attr_key] = len(attribute_set)  # Assign next available index
                mapping[node_type].append(attribute_set[attr_key])
    
    # Create the final ordered representation
    vector_representation = list(attribute_set.keys())
    
    return vector_representation, mapping

def extract_nested_list(input_string: str):
    start = input_string.find("[")+1
    end = input_string.find("]", start)
    if start != 0 and end != -1:
        return input_string[start:end]
    return ""

def extract_node_type(input_string: str):
    start = input_string.find("(") + 1
    end = input_string.find(",", start)
    if start != 0 and end != -1:
        return input_string[start:end]
    return ""

# Example usage
node_details = """
A: Float, Int, String
B: Float, String, Int
D: Enum1
C: String, Int, Bool, Int
E: 
"""
enum_details = """
Enum1 - 4
"""
vector_rep, mapping = generate_vector_representation(node_details, enum_details)


# Example extraction usage
new_compiled_list = []
# input_string = """pset.addPrimitive(LazyConv2d, 
#                   [Tensor3D, ChannelSize, KernelSize, KernelSize, StrideSize, StrideSize, PaddingSize, PaddingSize, PaddingMode, DilationSize, DilationSize, GroupSize], 
#                   Tensor3D)"""
big_input = ""
with open("/home/hice1/psomu3/scratch/surrogate-evolution/primitives copy.txt", 'r') as f:
    big_input = f.read()
for input_string in big_input.split("\n\n"):
    node = extract_node_type(input_string)
    extracted_list = extract_nested_list(input_string)
    parts = extracted_list.split(", ")
    # print(node, parts)
    newStr = ""
    if parts[0] == 'Tensor3D':
        newStr = node + ": " + ", ".join(parts[1:])
    else:
        newStr = node + ": " + ", ".join(parts)
    print("ADDING", newStr)
    new_compiled_list.append(newStr)
new_compiled = "\n".join(new_compiled_list)

enum_details = """
UpsampleMode - 5
ConvNeXtSize - 4
DenseNetSize - 4
EfficientNet_V2Size - 3
MobileNet_V3Size - 2
ShuffleNet_V2Size - 4
Swin_V2Size - 3
BoolWeight - 2
RegNetSize - 7
ResNeXtSize - 2
ResNetSize - 3
Wide_ResNetSize - 2
Weights - 3
PaddingMode - 4
CyclicLRMode - 3
CyclicLRScaleMode - 2
AnnealStrategy - 2
"""
print(new_compiled)
vector_rep, mapping = generate_vector_representation(new_compiled, enum_details)
print("Vectorized Representation:", vector_rep)
print(len(vector_rep))
print("Mapping:", mapping)