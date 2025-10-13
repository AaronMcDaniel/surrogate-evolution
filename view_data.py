import torch

# Load the data file
data = torch.load('/home/hice1/wlu314/scratch/surrogate-evolution/data.pt')

# Print basic information
print(f"Type of loaded data: {type(data)}")

# If it's a dictionary, print the keys
if isinstance(data, dict):
    print(f"Keys in the data: {data.keys()}")
    
    # Print a sample of each chunk/key
    for key, value in data.items():
        print(f"\nKey: {key}")
        print(f"Type: {type(value)}")
        
        # For tensors
        if torch.is_tensor(value):
            print(f"Shape: {value.shape}")
            print(f"Data type: {value.dtype}")
            print(f"Sample data (first few elements):")
            print(value.flatten()[:10])  # Print first 10 elements
        # For lists
        elif isinstance(value, list):
            print(f"Length: {len(value)}")
            print(f"Sample: {value[:3]}")  # Print first 3 elements
        # For other types
        else:
            print(f"Value: {value}")