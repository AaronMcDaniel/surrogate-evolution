# Imports
import torch
import torch.nn as nn
from torch.fx import Tracer, GraphModule
import torchvision.models as models
import torchdata
import torch_geometric
import pickle
import shlex
import sys
from torch.utils.data import DataLoader, Dataset
import os
import gc
import re
import torch.utils.data.dataset
import glob
import pandas as pd
import numpy as np
from collections import OrderedDict
from codec import Codec #Only import from code-base, need to have a sub-repo etc.
import torch.nn.functional as F
from torch_geometric.data import HeteroData, Data
import networkx as nx
import plotly.graph_objects as go
from torch_geometric.utils import to_networkx

DEVICE = torch.device("cpu")


"""Names of all heads, schedulers, and optimizers"""

head_classes = ["FasterRCNN_Head", "FCOS_Head", "RetinaNet_Head",'SSD_Head']
optimizer_classes = ['Adam', 'Adamax', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'NAdam','RAdam', 'AdamW', 'Rprop', 'ASGD']
scheduler_classes = ['StepLR', 'ExponentialLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau', 'CyclicLR', 'OneCycleLR', 'MultiStepLR', 'ConstantLR', 'PolynomialLR','LinearLR']


"""
Creates a dictionary of all the modules in the model, where the key is the 
name used in the symbolically traced graph and the value is the actual module.
"""
def create_model_dict(model):
    model_dict = {}
    #Empty list represents prefixes, of which there are none rn
    _create_model_dictH(model,[],model_dict)
    return model_dict
    #Gonna  need to create another function to go from Conv2d primitive to something else, idk
"""
Helper method for create_model_dict
"""
def _create_model_dictH(current,prefixes,model_dict):
    if len(current._modules) == 0:
        #As far as can go, basic torch primitive.
        name = "_".join(prefixes)
        model_dict[name] = current
    else:
        #Not a primitive, loop through its modules
        for prefix, module in current._modules.items():
            _create_model_dictH(module,prefixes+[prefix],model_dict)

"""Returns a set of all the different prim types used in the model. This will be needed for the one-hot-encoded representation."""

def get_prim_types(model):
    model_dict = create_model_dict(model)
    types = set()
    for val in model_dict.values():
        types.add(type(val))
    return types


"""
Custom Symbolic Tracer to fix NameError, module not installed as submodule, by instead giving it the name "unregistered_module"
"""
# Custom Tracer to handle missing submodules
class CustomTracer(Tracer):
    def path_of_module(self, mod):
        # Try to get the path of the module as usual
        try:
            return super().path_of_module(mod)
        except NameError:
            # Handle unregistered modules by assigning them a unique name
            module_name = f"unregistered_module_{id(mod)}"
            self.root.add_module(module_name, mod)  # Register it dynamically
            return module_name


# Define the tracing function using the custom tracer
def symbolic_trace_with_custom_tracer(model, concrete_args=None):
    tracer = CustomTracer()
    graph = tracer.trace(model, concrete_args)
    return GraphModule(tracer.root, graph, model.__class__.__name__)


"""Function to parse extr_repr params into args and kwargs
    '3, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False'
 -> ([3, 24],
 {'kernel_size': (3, 3), 'stride': (2, 2), 'padding': (1, 1), 'bias': False})
"""

def parse_representation(repr_str):
    args = []
    kwargs = {}

    # Split positional and keyword arguments based on '=' for named args
    parts = re.split(r',\s*(?![^()]*\))', repr_str)

    for part in parts:
        if '=' in part:
            # Process keyword arguments
            key, value = part.split('=', 1)
            key = key.strip()

            # Use eval to safely convert values like False, tuples, etc.
            try:
                value = eval(value.strip())
            except:
                value = value.strip()  # Leave it as a string if eval fails

            kwargs[key] = value
        else:
            # Process positional arguments
            try:
                args.append(eval(part.strip()))
            except:
                args.append(part.strip())  # Leave as string if eval fails

    return args, kwargs

"""Function to encoder torch modules (The ones identified by create_model_dict) into feature vectors.
This is dynamically done based on the present params. We may not use this method of encoding due to it resulting
in a large number of different node types. 
"""

def encode_module_repr_1(module):
    params_repr = module.extra_repr()
    result = parse_representation(params_repr)
    #I should make this more complicated in the future, but for now this will just be this.\
    #Each different module will have its own type
    #for key, val in kwargs.
    encoding = []
    for key,val in result.items():
        if key == "eps": #Not sure if we should do this, we will see
            #add log
            encoding.append(np.log(val))
        elif val == "True":
            encoding.append(1)
        elif val == "False":
            encoding.append(0)
        #Tuple Input
        elif isinstance(val,tuple) or isinstance(val,list):
            encoding.extend(list(val))
        else:
            try:
                encoding.append(float(val))
            except:
                pass
                #If nothing works, simply don't include it
    return torch.tensor(encoding)


"""
Alternate representation, where we only have one node-type for params, and use a one-hot encoding of the modules instead.
Must pass in set of all types used.
"""

def encode_module_repr_2(module,set_types):
    vocab_to_index = {word: idx for idx, word in enumerate(set_types)}
    num_types = len(set_types) + 1
    mod_type = type(module)
    if mod_type in vocab_to_index:
        idx = vocab_to_index[mod_type] + 1
    else:
        idx = 0
    encoding = [0] * num_types
    encoding[idx] = 1
    return encoding


'''
New Encoding Strategy for GNN
Dense Vector (Module Type) + Hyperparameters (of the Module)
'''

## Universal template from `module_hyperparameters.py`
universal_template = [
    '_random_samples', 'affine', 'align_corners', 'approximate', 'args', 'bias', 'ceil_mode', 
    'count_include_pad', 'device', 'dilation', 'dims', 'divisor_override', 'dtype', 
    'elementwise_affine', 'eps', 'groups', 'in_channels', 'in_features', 'inplace', 'kernel_size', 
    'kwargs', 'mode', 'momentum', 'negative_slope', 'norm_type', 'normalized_shape', 'num_features', 
    'out_channels', 'out_features', 'output_padding', 'output_ratio', 'output_size', 'p', 'padding', 
    'padding_mode', 'recompute_scale_factor', 'return_indices', 'scale_factor', 'size', 'stride', 
    'threshold', 'track_running_stats', 'value'
]

def encode_module_hyperparams_v1(module, universal_template):
    """
    - Encodes the hyperparameters of a module into a fixed-length vector
    
    - module.extra_repr() to get the parameter string, then parses it,
    and finally maps the available parameters to positions in the universal template

    - missing parameter gets a default value of zero
    """
    vector = [0.0] * len(universal_template)
    try:
        _, kwargs = parse_representation(module.extra_repr())
    except Exception as e:
        return torch.tensor(vector, dtype=torch.float)
    
    # loop over each parameter
    # if it exists, convert it to float and store it
    for i, key in enumerate(universal_template):
        if key in kwargs:
            try:
                vector[i] = float(kwargs[key])
            except Exception:
                vector[i] = 0.0
        else:
            vector[i] = 0.0  # missing parameters

    return torch.tensor(vector, dtype=torch.float)

def encode_module_index(module, set_types):
    """
    `encode_module_repr_2`, creates a one‐hot vector based on the module type. 
    However, for a learned embedding, you need a function that simply returns an integer index.
    This function is conceptually similar to encode_module_repr_2 but returns an index (an integer) rather than a one‐hot vector.
    """
    vocab_to_index = {word: idx for idx, word in enumerate(set_types)}
    mod_type = type(module)
    return vocab_to_index.get(mod_type, 0)

"""
Graph representation using the new encoding strategy
Dense Vector + Hyperparameters
"""
def create_graph_repr_custom(backbone, set_types, encoding_strategy="regular"):
    """
    Convert the backbone into a PyG Data object using one of two encoding strategies:
    
    - If encoding_strategy == "regular", each node is represented only by its module type
      (stored as an integer index to be fed to an embedding layer later).
    
    - If encoding_strategy == "combined", each node is represented by its module type index
      and a fixed-length hyperparameter vector, based on a universal hyperparameter template.
    
    Returns:
        datum: A torch_geometric.data.Data object.
        node_index_to_name: A mapping from node indices to descriptive names.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = backbone.to(DEVICE)
    model_dict = create_model_dict(backbone)
    model_graph = symbolic_trace_with_custom_tracer(backbone)
    nodes = model_graph.graph.nodes

    if encoding_strategy == "regular":
        # Strategy: Only use the module type index.
        for node in nodes:
            module_instance = model_dict.get(node.name)
            # Get the type index (using your existing function)
            node.module_index = encode_module_index(module_instance, set_types)
            if node.name in model_dict:
                node.descriptive_name = model_dict[node.name]._get_name()
            else:
                node.descriptive_name = node.name
        # Build tensor of module type indices
        node_features = torch.tensor([node.module_index for node in nodes], dtype=torch.long)
    
    elif encoding_strategy == "combined":
        universal_template = [
            '_random_samples', 'affine', 'align_corners', 'approximate', 'args', 'bias', 'ceil_mode', 
            'count_include_pad', 'device', 'dilation', 'dims', 'divisor_override', 'dtype', 
            'elementwise_affine', 'eps', 'groups', 'in_channels', 'in_features', 'inplace', 'kernel_size', 
            'kwargs', 'mode', 'momentum', 'negative_slope', 'norm_type', 'normalized_shape', 'num_features', 
            'out_channels', 'out_features', 'output_padding', 'output_ratio', 'output_size', 'p', 'padding', 
            'padding_mode', 'recompute_scale_factor', 'return_indices', 'scale_factor', 'size', 'stride', 
            'threshold', 'track_running_stats', 'value'
        ]
        for node in nodes:
            module_instance = model_dict.get(node.name)
            node.module_index = encode_module_index(module_instance, set_types)
            node.hyperparams = encode_module_hyperparams_v1(module_instance, universal_template)
            if node.name in model_dict:
                node.descriptive_name = model_dict[node.name]._get_name()
            else:
                node.descriptive_name = node.name
        # Build tensor for module type indices, hyperparmeter vector
        type_indices = torch.tensor([node.module_index for node in nodes], dtype=torch.long)
        hyperparams_tensor = torch.stack([node.hyperparams for node in nodes])
        node_features = type_indices
    else:
        raise ValueError("Invalid encoding_strategy. Choose either 'type' or 'combined'.")

    # Build edge list from the traced graph.
    node_name_to_index = {node.name: idx for idx, node in enumerate(nodes)}
    edge_list = []
    for node in nodes:
        src_id = node_name_to_index[node.name]
        for target in node.users.keys():
            tgt_id = node_name_to_index[target.name]
            edge_list.append([src_id, tgt_id])
    edge_index = torch.tensor(edge_list).T.contiguous()

    datum = Data(x=node_features, edge_index=edge_index)
    
    # For the "combined" strategy attach hyperparameters as a custom attribute.
    if encoding_strategy == "combined":
        datum.hyperparams = hyperparams_tensor

    # Create a mapping from node index to descriptive name.
    node_index_to_name = {idx: node.descriptive_name for idx, node in enumerate(nodes)}
    return datum, node_index_to_name



"""Convert given backbone into a Torch Geometric data object"""
def create_graph_repr(backbone,set_types):
   
    backbone = backbone.to(DEVICE)

    #Is testing the model once necessary? Sometimes the dummy_input isn't of the right shape
    #dummy_input = torch.randn((1,3,1000,1000)).to(device) #Will need to find what the proper input size
    #backbone(dummy_input) #Carry out one iteration to ensure it works
    #May not even be needed ^^

    #Encode the model_info and pass in both as a tuple.
    model_dict = create_model_dict(backbone)

    model_graph = symbolic_trace_with_custom_tracer(backbone)

    nodes = model_graph.graph.nodes
    #Temporary, eventually replace with set_type for all models so its consistent
    for node in nodes:
        node.value = encode_module_repr_2(model_dict.get(node.name),set_types)
        if node.name in model_dict:
            node.descriptive_name = model_dict[node.name]._get_name()
        else:
            node.descriptive_name = node.name
    
    #Get id for each name
    node_name_to_index = {node.name: idx for idx, node in enumerate(nodes)}
    node_index_to_name = {idx: node.descriptive_name for idx , node in enumerate(nodes)}
    #Create edge list
    edge_list = []
    for node in nodes:
        src_id = node_name_to_index[node.name]
        for target in node.users.keys():
            tgt_id = node_name_to_index[target.name]
            edge_list.append([src_id,tgt_id])

    #Get features vector
    node_features = torch.tensor([node.value for node in nodes], dtype=torch.float)
    edge_index = torch.tensor(edge_list).T.contiguous()

    datum = Data(x=node_features, edge_index=edge_index)

    return datum,node_index_to_name


#One-hot encoding
def encoder(types,name):
  idx = types.index(name)
  one_hot = F.one_hot(torch.tensor(idx), num_classes=len(types))
  return one_hot
#Not one-hot encoded features
#There are certain scheduler specific parameters that haven't been included, because they depend on which scheduler is used
def feature_vector(input_data):
    optimizer_lr = torch.tensor(input_data['optimizer_lr'])
    # optimizer_weight_decay = torch.tensor(input_data['optimizer_weight_decay'])
    loss_weights = input_data['loss_weights']
    feature_vector = torch.cat([
        optimizer_lr.unsqueeze(0),
        # optimizer_weight_decay.unsqueeze(0),
        loss_weights
    ])

    return feature_vector

def vectorize_model_info(model_info):
    optimizer = encoder(optimizer_classes,model_info["optimizer"])
    head = encoder(head_classes,model_info["head"])
    scheduler = encoder(scheduler_classes,model_info["lr_scheduler"])
    feature = feature_vector(model_info)

    #Take the log and then normalize, should look into better ways to do this.
    feature = torch.log(feature + 1e-6) #Small eps if 0
    mu = torch.mean(feature)
    std  = torch.std(feature)
    feature = (feature - mu ) / std
    information = torch.cat((optimizer,head,scheduler,feature))
    return information


"""Convert a genome into torch geometric data object"""
def create_data_object(genome,set_types):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = Codec(num_classes=7,genome_encoding_strat='Tree')
    model_info = codec.decode_genome(genome,num_loss_components=7)
    model = model_info["model"]
    del model_info["model"]
    backbone = model._modules["backbone"].to(device)
    model_info["head"] = genome.split("(")[0]
    model_info_vector = vectorize_model_info(model_info)
    #datum, index_to_name = create_graph_repr(backbone,set_types)
    datum, index_to_name = create_graph_repr_custom(backbone, set_types, encoding_strategy="combined") #change to combined encdoing

    return model_info_vector, datum, index_to_name


class SurrogateData(Dataset):
    def __init__(self,csv_path,set_types_pkl,selected_metrics):
        #selected metrics is the list of desired metrics from the csv file, and the order they should be returned
        self.selected_metrics =selected_metrics
        self.dataframe = pd.read_csv(csv_path)
        for metric in self.selected_metrics:
            if metric not in self.dataframe.columns:
                raise Exception(f"{metric} is not an actual metric")
        with open(set_types_pkl, "rb") as f:
            self.set_types = pickle.load(f)
        self.cache = [None for _ in range(self.__len__())]
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self,idx):
        if self.cache[idx] == None:

            genome = self.dataframe["genome"][idx]
            label = torch.tensor(self.dataframe.loc[idx,self.selected_metrics])

            model_info_vector, datum, _ = create_data_object(genome,self.set_types)

            self.cache[idx] = model_info_vector, datum, label
        else:
            model_info_vector, datum, label = self.cache[idx]
        return model_info_vector, datum, label

def visualize_interactive_graph(data, node_names=None):
    """
    Visualize a torch_geometric.data.Data object with node names in an interactive graph.
    
    Args:
        data: The torch_geometric.data.Data object.
        node_names: A dictionary mapping node indices to node names.
                   If None, it will default to showing node indices.
    """
    # Convert torch_geometric.data.Data to a directed NetworkX graph
    G = to_networkx(data, to_undirected=False)

    # Get node positions using the spring layout (or any other layout you like)
    pos = nx.spring_layout(G)

    # Extract edge traces (with arrows)
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                mode='lines'))

    # Extract node traces (with hover text)
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
        )
    )

    # Fill node positions and text information
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        
        # If node names are provided, use them. Otherwise, fall back to the node index
        node_label = f"Node {node}"
        if node_names and node in node_names:
            node_label = f"{node_names[node]} (Node {node})"
        
        node_trace['text'] += tuple([node_label])
        node_trace['marker']['color'] += tuple([len(G[node])])  # Color by node degree

    # Create the plot
    fig = go.Figure(
        data=edge_trace + [node_trace],
        layout=go.Layout(
            title='<br>Interactive Directed Graph',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            annotations=[dict(
                text="Graph Visualization",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 )],
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False))
    )

    # Show the plot in the browser
    fig.show()

    
def generate_data_withencoding_pt(
    csv_path,
    set_types_pkl,
    selected_metrics,
    output_file
):
    """
    Like generate_data_withencoding_pt, but logs memory before/after each step.
    If debug_limit is set, stops after that many samples.
    """
    dataframe = pd.read_csv(csv_path)  # compiled_data_valid_only.csv

    # loading set_types
    with open(set_types_pkl, "rb") as f:
        set_types = pickle.load(f)

    data_dict = {}
    for i in range(len(dataframe)):
        genome = dataframe["genome"][i]

        # build label as a float tensor, safe against object dtypes
        vals = []
        for m in selected_metrics:
            try:
                vals.append(float(dataframe.loc[i, m]))
            except Exception:
                vals.append(0.0)
        label = torch.tensor(vals, dtype=torch.float)

        model_info_vector, datum, _ = create_data_object(genome, set_types)
        data_dict[i] = (model_info_vector, datum, label)

        if i % 100 == 0:
            print(f"Processed {i} samples...")

    # save the data
    torch.save(data_dict, output_file)
    print(f"Saved dataset with {len(dataframe)} samples to {output_file}")

        
def generate_data_pt(csv_path, set_types_pkl, selected_metrics, output_file="data.pt"):
    """
    Generates a new data.pt file using the current SurrogateData dataset
    
    This should include the hyperparams
    
    Parameters:
      csv_path (str): Path to the CSV file (e.g. compiled_data_valid_only.csv).
      set_types_pkl (str): Path to the pickle file containing the set of module types.
      selected_metrics (list): List of metric names (e.g. ["iou_loss", "giou_loss", "ciou_loss"]).
      output_file (str): Name of the output file to save the dataset.
      
    Returns:
      None
    """
    dataset = SurrogateData(csv_path, set_types_pkl, selected_metrics)
    data_dict = {}
    for i in range(len(dataset)):
        # Each sample is a tuple: (model_info_vector, datum, label)
        data_dict[i] = dataset[i]
        if i % 100 == 0:
            print(f"Processed {i} samples...")
    torch.save(data_dict, output_file)
    print(f"Saved dataset with {len(dataset)} samples to {output_file}")



## Run this to generate the data_with_encoding.pt file
if __name__ == "__main__":
    csv_path = "/home/hice1/wlu314/scratch/surrogate-evolution/preprocess_data/data_51.csv"
    set_types_pkl = "./set_prims.pkl"
    selected_metrics = ["iou_loss", "giou_loss", "ciou_loss"]
    generate_data_withencoding_pt(
        csv_path,
        set_types_pkl,
        selected_metrics,
        output_file="/home/hice1/wlu314/scratch/surrogate-evolution/preprocess_data/data_with_encoding_51.pt",
    )


# Run this to generate the data_hyperparams.pt file
# if __name__ == "__main__":
#     csv_path = "./data/compiled_data_valid_only.csv"
#     set_types_pkl = "./set_prims.pkl"
#     selected_metrics = ["iou_loss", "giou_loss", "ciou_loss"]
#     generate_data_pt(csv_path, set_types_pkl, selected_metrics, output_file="data_hyperparams.pt")


    
# if __name__ == "__main__":    
#     num_classes = 7
#     num_loss_components = 7
#     codec = Codec(num_classes=num_classes)
#     counter = 0

#     data_path = "./data/compiled_data_valid_only.csv"
#     df = pd.read_csv(data_path)


#     #Just temporary code for getting the set of everything, deal with this more formally later
#     def process_genome(genome):
#         model_dict = codec.decode_genome(genome, num_loss_components)
#         model = model_dict['model']
#         return model

#     def get_prim_types(model):
#         model = model._modules["backbone"].to("cuda")
#         model_dict = create_model_dict(model)
#         types = set(type(val) for val in model_dict.values())
#         return types

#     df['processed_genome'] = df['genome'].apply(process_genome)

#     all_types = set()

#     for model in df['processed_genome']:
#         counter += 1
#         print(counter)
#         try:
#             model_types = get_prim_types(model)
#             all_types.update(model_types)
#         except:
#             print(f"{counter} failed...")
#     print("saving")
#     output_path = "./data/all_unique_types.txt"
#     with open(output_path, 'w') as f:
#         for t in all_types:
#             f.write(f"{t}\n")
            
