import pickle
import inspect
import torch
from collections import Counter

# Load the modules from the pickle file
with open("set_prims.pkl", "rb") as f:
    set_prims = pickle.load(f)
    
print("This contains all the modules within set_prims.pkl:")
print(set_prims)

# Using a set to build a universal template of hyperparameter names
all_hyperparameters = set()
# Using a Counter to count the frequency of each hyperparameter
hyperparameter_counts = Counter()

for module in set_prims:
    # Attempt to get a readable module name
    module_name = getattr(module, '__name__', str(module))
    print(f"Processing module: {module_name}")
    try: 
        signature = inspect.signature(module.__init__)
        # Iterate over all parameters in the __init__ signature
        for name, param in signature.parameters.items():
            if name != "self":
                all_hyperparameters.add(name)
                hyperparameter_counts[name] += 1
    except Exception as e:
        print(f"Error processing {module_name}: {e}")

# Create a universal hyperparameter template (sorted alphabetically)
universal_template = sorted(list(all_hyperparameters))
print("Universal Hyperparameter Template:")
print(universal_template)
print("This is the size of the hyperparameter template:")
print(len(universal_template))

# Sort the hyperparameter frequency dictionary from most frequent to least
sorted_hyperparameter_counts = dict(sorted(hyperparameter_counts.items(), key=lambda item: item[1], reverse=True))
print("Sorted Hyperparameter Frequency Dictionary (from most to least frequent):")
print(sorted_hyperparameter_counts)
