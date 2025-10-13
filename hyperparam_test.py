from data_handling_test import create_data_object, visualize_interactive_graph
import torch
import pickle


test_genome = "RetinaNet_Head(ShuffleNet_V2(IN0, 1, dummyOp(dummyOp(1))), Adamax(add(...)))"

with open("set_prims.pkl", "rb") as f:
    set_types = pickle.load(f)
    
print(set_types)

# Test with hyperparameters included:
model_info_vector, datum, node_names = create_data_object(test_genome, set_types, includeHyperparams=True)
print("Model Info Vector Shape:", model_info_vector.shape)
print("Graph Node Feature Matrix Shape:", datum.x.shape)

