import torch

# Load the data file - update the path to your data_with_encoding_chunks file
# Assuming it has a .pt extension, if not, adjust accordingly
data = torch.load('data_with_encoding_all.pt')

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
            
            # If this is a list of chunks, you might want more details about each chunk
            if len(value) > 0:
                print("\nFirst chunk details:")
                chunk = value[0]
                print(f"Chunk type: {type(chunk)}")
                if torch.is_tensor(chunk):
                    print(f"Chunk shape: {chunk.shape}")
                    print(f"Chunk sample: {chunk.flatten()[:5]}")
        # For other types
        else:
            print(f"Value: {value}") 

