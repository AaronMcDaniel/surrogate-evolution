import os
import pickle
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from itertools import product


import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd

## DIFFERENT NETWORKS
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.nn import global_max_pool

import torch.optim as optim
from data_handling import SurrogateData
import matplotlib.pyplot as plt

data_path = "/home/hice1/wlu314/scratch/surrogate-evolution/data/compiled_data_valid_only.csv"
set_prims_path = "/home/hice1/wlu314/scratch/surrogate-evolution/set_prims.pkl"
selected_metrics = ["iou_loss", "giou_loss", "ciou_loss"]

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = list(data.values()) #Cuz it might not have every one taken.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
#Pre_processed data rn
data = torch.load("data.pt")
dataset = MyDataset(data)

first_data = dataset[0]

vector, graph_data, label = first_data

print("Vector:", vector)
print("Vector Shape:", vector.shape)
print("Vector Type:", type(vector))

print("Graph Data:", graph_data)
print("Graph Data Type:", type(graph_data))

print("Node Features (x):", graph_data.x)
print("Node Features Shape:", graph_data.x.shape)
print("Node Features Type:", type(graph_data.x))

print("Edge Index:", graph_data.edge_index)
print("Edge Index Shape:", graph_data.edge_index.shape)
print("Edge Index Type:", type(graph_data.edge_index))

print("Label:", label)
print("Label Shape:", label.shape)
print("Label Type:", type(label))

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print("Split dataset into training and testing sets")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


## Load the data with encoding
data_withEncoding = torch.load("data_with_encoding_all.pt")
dataset_withEncoding = MyDataset(data_withEncoding)

# Create train/test split for the encoded data
train_size_encoded = int(0.8 * len(dataset_withEncoding))
test_size_encoded = len(dataset_withEncoding) - train_size_encoded
train_dataset_encoded, test_dataset_encoded = torch.utils.data.random_split(
    dataset_withEncoding, [train_size_encoded, test_size_encoded]
)

# Create data loaders for the encoded data
train_loader_encoded = DataLoader(train_dataset_encoded, batch_size=32, shuffle=True)
test_loader_encoded = DataLoader(test_dataset_encoded, batch_size=32, shuffle=False)

#Try it with the MLP and other encoding format

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pooling to get a graph-level representation
        #x = torch.zeros_like(x)
        x = self.linear(x)
        return x

class SimpleGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=8):
        super(SimpleGAT, self).__init__()
        # First GAT layer with multiple heads; concatenates the head outputs
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        # Second GAT layer: here we set heads=1 and disable concatenation so the output dimension remains hidden_dim
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Pooling for a graph-level representation
        x = self.linear(x)
        return x
    



class SimpleGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGIN, self).__init__()
        # Define an MLP for the first GIN layer
        mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(mlp1)
        
        # Define an MLP for the second GIN layer
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv2 = GINConv(mlp2)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

class SimpleGraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x

## Advanced GNN Models
 ## Multiple Convulution Layers 
 ## Batch normalization and dropout for regulation
 ## Residual connection (for GATs)
class AdvancedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.5):
        super(AdvancedGCN, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # first GCN layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # last GCN layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Readout: concatenate global mean and max pooling
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply BN, activation and dropout for all but last layer
            if i < len(self.convs) - 1:
                x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        # Pooling: combine both mean and max pool for richer representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


"""
This is for the new encoding
"""
class CombinedFeatureGCN(nn.Module):
    def __init__(self, num_module_types, type_embedding_dim, hyperparam_dim, hidden_dim, output_dim):
        super(CombinedFeatureGCN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_module_types, embedding_dim=type_embedding_dim)
        # Input to the first GCN layer is the concatenation of the type embedding and the hyperparameter vector.
        self.conv1 = GCNConv(type_embedding_dim + hyperparam_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # data.x contains the module type indices to a dense vector representation
        # data.hyperparams contains the fixed-length hyperparameter vector (currently 43)
        type_embeds = self.embedding(data.x.squeeze()) 
        x = torch.cat([type_embeds, data.hyperparams], dim=1)
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.linear(x)
        return x

def testCombinedFeatureGCN(
    num_module_types, 
    type_embedding_dim, 
    hyperparam_dim, 
    hidden_dim, 
    output_dim, 
    epoch=25
):
    """
    Train and test the CombinedFeatureGCN model using encoded data.
    Returns:
      model: the trained model
      avg_iou:  IOU‐loss
      avg_giou: GIOU‐loss
      avg_ciou: CIOU‐loss
    """
    model     = CombinedFeatureGCN(num_module_types, type_embedding_dim,
                                   hyperparam_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for ep in range(epoch):
        model.train()
        total_loss = 0.0
        total_iou_loss = 0.0
        total_giou_loss = 0.0
        total_ciou_loss = 0.0
        for vector, data, labels in train_loader_encoded:  # Use encoded data loader
            # Ensure proper types.
            data.x = data.x.long()
            if hasattr(data, 'hyperparams'):
                data.hyperparams = data.hyperparams.float()
            labels = labels.float()
            
            optimizer.zero_grad()
            outputs = model(data)  # outputs shape: [batch_size, 3]
            # Compute individual losses for each metric
            loss_iou = criterion(outputs[:, 0], labels[:, 0])
            loss_giou = criterion(outputs[:, 1], labels[:, 1])
            loss_ciou = criterion(outputs[:, 2], labels[:, 2])
            
            # You can weight these losses differently if desired
            total_batch_loss = loss_iou + loss_giou + loss_ciou
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_iou_loss += loss_iou.item()
            total_giou_loss += loss_giou.item()
            total_ciou_loss += loss_ciou.item()
            
        avg_loss = total_loss / len(train_loader_encoded)
        avg_iou = total_iou_loss / len(train_loader_encoded)
        avg_giou = total_giou_loss / len(train_loader_encoded)
        avg_ciou = total_ciou_loss / len(train_loader_encoded)
        print(f"Epoch {ep+1}/{epoch} - Total Loss: {avg_loss:.4f}, IOU: {avg_iou:.4f}, GIOU: {avg_giou:.4f}, CIOU: {avg_ciou:.4f}")

    # Evaluation loop on test data
    model.eval()
    total_iou, total_giou, total_ciou = 0.0, 0.0, 0.0

    with torch.no_grad():
        for _, data, labels in test_loader_encoded:
            data.x = data.x.long()
            if hasattr(data, 'hyperparams'):
                data.hyperparams = data.hyperparams.float()
            labels = labels.float()

            out = model(data)
            total_iou  += criterion(out[:, 0], labels[:, 0]).item()
            total_giou += criterion(out[:, 1], labels[:, 1]).item()
            total_ciou += criterion(out[:, 2], labels[:, 2]).item()

    n = len(test_loader_encoded)
    avg_iou, avg_giou, avg_ciou = total_iou/n, total_giou/n, total_ciou/n

    print(f"Final Test IOU Loss:  {avg_iou:.4f}")
    print(f"Final Test GIOU Loss: {avg_giou:.4f}")
    print(f"Final Test CIOU Loss: {avg_ciou:.4f}")

    return model, avg_iou, avg_giou, avg_ciou


def testAdvancedGCN(
    input_dim=28, 
    hidden_dim=64, 
    output_dim=3, 
    num_layers=4, 
    dropout=0.5, 
    num_epochs=25, 
    lr=0.01
):
    """
    Train and test the AdvancedGCN model.
    
    Parameters:
      input_dim (int): Number of node features.
      hidden_dim (int): Hidden dimension for GCN layers.
      output_dim (int): Number of output features.
      num_layers (int): Number of GCN layers in AdvancedGCN.
      dropout (float): Dropout rate.
      num_epochs (int): Number of training epochs.
      lr (float): Learning rate.
    
    Returns:
      model: The trained AdvancedGCN model.
      train_losses (list): List of training losses per epoch.
      test_losses (list): List of testing losses per epoch.
    """
    # Instantiate the AdvancedGCN model
    model = AdvancedGCN(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    # Training and testing loop
    for ep in range(num_epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        test_loss = test_epoch(model, criterion, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"AdvancedGCN Epoch {ep+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Plotting the loss curves
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss (MSE)")
    plt.plot(range(1, num_epochs+1), test_losses, label="Test Loss (MSE)")
    plt.title("AdvancedGCN Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("AdvancedGCN_loss.png")
    plt.show()

    return model, train_losses, test_losses






input_dim = 28  # Number of node features
output_dim = 3  # Number of output features (based on labels)

## Change to a function
def train_epoch(model, optimizer, criterion, loader):
    model.train()
    total_loss = 0
    for vector, data, labels in loader:
        data.x = data.x.float()
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test_epoch(model, criterion, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for vector, data, labels in loader:
            data.x = data.x.float()
            labels = labels.float()
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

def run_experiment(model_class, variant_name, hidden_dim, lr, num_epochs, heads=None):
    if variant_name == "GAT":
        model = model_class(input_dim, hidden_dim, output_dim, heads=heads)
    else:
        model = model_class(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader)
        test_loss = test_epoch(model, criterion, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"{variant_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    # Return the model, the loss curves, and the final test loss
    return model, train_losses, test_losses, test_losses[-1] 



def run_all_experiments(
    variants, 
    learning_rates=[0.001, 0.01], 
    hidden_dims=[64, 128], 
    num_epochs=25, 
    heads_options=[8]
):
    """
    Run grid search experiments over multiple model variants and hyperparameters.
    
    Parameters:
      variants (dict): Dictionary mapping variant names (e.g., "GCN", "GAT", "GIN", "GraphSAGE") to model classes.
      learning_rates (list): List of learning rates to try.
      hidden_dims (list): List of hidden dimensions to try.
      num_epochs (int): Number of training epochs.
      heads_options (list): List of head options (only used for the GAT variant).
      
    Returns:
      results_df (pandas.DataFrame): A DataFrame summarizing the final test loss for each variant and hyperparameter combination.
    """

    results = []
    
    # Loop over each model variant and hyperparameter combination.
    for variant_name, model_class in variants.items():
        for lr in learning_rates:
            for hidden_dim in hidden_dims:
                print(f"\nRunning {variant_name} with lr={lr}, hidden_dim={hidden_dim}")
                if variant_name == "GAT":
                    for heads in heads_options:
                        model, train_losses, test_losses, final_test_loss = run_experiment(
                            model_class, variant_name, hidden_dim, lr, num_epochs, heads=heads
                        )
                        results.append({
                            'variant': variant_name,
                            'lr': lr,
                            'hidden_dim': hidden_dim,
                            'heads': heads,
                            'final_test_loss': final_test_loss
                        })
                        # Plot and save loss curves
                        epochs_range = list(range(1, num_epochs+1))
                        plt.figure()
                        plt.plot(epochs_range, train_losses, label="Train Loss (MSE)")
                        plt.plot(epochs_range, test_losses, label="Test Loss (MSE)")
                        plt.title(f"{variant_name} (lr={lr}, hidden_dim={hidden_dim}, heads={heads})")
                        plt.ylabel("MSE Loss")
                        plt.xlabel("Epoch")
                        plt.legend()
                        plt.savefig(f"{variant_name}_lr{lr}_hd{hidden_dim}_heads{heads}_loss.png")
                        plt.close()
                else:
                    model, train_losses, test_losses, final_test_loss = run_experiment(
                        model_class, variant_name, hidden_dim, lr, num_epochs
                    )
                    results.append({
                        'variant': variant_name,
                        'lr': lr,
                        'hidden_dim': hidden_dim,
                        'final_test_loss': final_test_loss
                    })
                    epochs_range = list(range(1, num_epochs+1))
                    plt.figure()
                    plt.plot(epochs_range, train_losses, label="Train Loss (MSE)")
                    plt.plot(epochs_range, test_losses, label="Test Loss (MSE)")
                    plt.title(f"{variant_name} (lr={lr}, hidden_dim={hidden_dim})")
                    plt.ylabel("MSE Loss")
                    plt.xlabel("Epoch")
                    plt.legend()
                    plt.savefig(f"{variant_name}_lr{lr}_hd{hidden_dim}_loss.png")
                    plt.close()

    results_df = pd.DataFrame(results)
    print("\nSummary of Results:")
    print(results_df)
    results_df.to_csv("experiment_results.csv", index=False)
    return results_df

variants = {
    "GCN": SimpleGCN,
    "GAT": SimpleGAT,
    "GIN": SimpleGIN,
    "GraphSAGE": SimpleGraphSAGE
}

#results_df = run_all_experiments(variants, learning_rates=[0.001, 0.01], hidden_dims=[64, 128], num_epochs=25, heads_options=[8])





def run_grid_search(variants, learning_rates=[0.001, 0.01], hidden_dims=[64, 128], num_epochs=25, heads_options=[8]):
    """
    Run grid search for multiple model variants over a range of hyperparameters.
    
    Parameters:
        variants (dict): Dictionary mapping variant names to model classes.
        learning_rates (list): List of learning rates to try.
        hidden_dims (list): List of hidden dimensions to try.
        num_epochs (int): Number of training epochs.
        heads_options (list): List of head options (only used for GAT variant).
    
    Returns:
        best_hyperparams (dict): Dictionary with best hyperparameters and final test loss for each variant.
        best_results_df (pandas.DataFrame): DataFrame summarizing the best hyperparameters and losses.
    """
    from itertools import product
    import pandas as pd

    best_hyperparams = {}

    def grid_search_for_variant(model_class, variant_name, learning_rates, hidden_dims, num_epochs, heads_options=None):
        best_loss = float('inf')
        best_params = {}
        # Iterate over all combinations of learning rate and hidden dimension.
        for lr, hidden_dim in product(learning_rates, hidden_dims):
            if variant_name == "GAT":
                # For GAT, iterate over heads as well.
                for heads in heads_options:
                    print(f"\nRunning {variant_name} with lr={lr}, hidden_dim={hidden_dim}, heads={heads}")
                    model, train_losses, test_losses, final_test_loss = run_experiment(
                        model_class, variant_name, hidden_dim, lr, num_epochs, heads=heads
                    )
                    if final_test_loss < best_loss:
                        best_loss = final_test_loss
                        best_params = {"lr": lr, "hidden_dim": hidden_dim, "heads": heads}
            else:
                print(f"\nRunning {variant_name} with lr={lr}, hidden_dim={hidden_dim}")
                model, train_losses, test_losses, final_test_loss = run_experiment(
                    model_class, variant_name, hidden_dim, lr, num_epochs
                )
                if final_test_loss < best_loss:
                    best_loss = final_test_loss
                    best_params = {"lr": lr, "hidden_dim": hidden_dim}
        return best_loss, best_params

    # Loop over each variant and perform grid search.
    for variant_name, model_class in variants.items():
        loss, params = grid_search_for_variant(
            model_class,
            variant_name,
            learning_rates,
            hidden_dims,
            num_epochs,
            heads_options=heads_options if variant_name == "GAT" else None
        )
        best_hyperparams[variant_name] = {"final_test_loss": loss, "params": params}
        print(f"\nBest hyperparameters for {variant_name}: {params} with loss: {loss}")

    # Create a DataFrame summarizing the best results.
    best_results_df = pd.DataFrame([
        {"variant": variant, **info["params"], "final_test_loss": info["final_test_loss"]}
        for variant, info in best_hyperparams.items()
    ])
    print("\nSummary of Best Hyperparameters:")
    print(best_results_df)

    # Save results to CSV
    best_results_df.to_csv("best_hyperparameters.csv", index=False)
    
    return best_hyperparams, best_results_df

#best_hyperparams, best_results_df = run_grid_search(variants)


# print("Creating model")
# model = SimpleGCN(input_dim, hidden_dim, output_dim)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.MSELoss()

# def train():
#     model.train()
#     total_loss = 0
#     for vector, data, labels in train_loader:
#         data.x = data.x.float()
#         labels = labels.float()
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(train_loader)

# def test():
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for vector, data, labels in test_loader:
#             data.x = data.x.float()
#             labels = labels.float()
#             outputs = model(data)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#     return total_loss / len(test_loader)

# print("Training model")
# num_epochs = 25
# train_losses = []
# test_losses = []
# for epoch in range(num_epochs):
#     train_loss = train()
#     test_loss = test()
#     train_losses.append(train_loss)
#     test_losses.append(test_loss)
#     print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


# epochs = [i for i in range(1,num_epochs+1)]

# plt.plot(epochs,train_losses,label="Train Loss (MSE)")
# plt.plot(epochs,test_losses,label="Test Loss (MSE)")
# plt.title("Losses of GCN surrogate")
# plt.ylabel("MSE Loss")
# plt.xlabel("Epoch")
# plt.legend()
# plt.savefig("example_fig.png")
# print("Training complete!")



if __name__ == "__main__":
    # 1) Load the encoded dataset
    data_enc = torch.load("data_with_encoding_all.pt")
    
    # 2) Compute number of module types
    #    data_enc.values() are tuples (vector, graph_data, label)
    all_types = torch.cat([
        graph.x.view(-1) 
        for (_, graph, _) in data_enc.values()
    ])
    num_module_types = int(all_types.max().item() + 1)
    
    # 3) Compute hyperparam_dim from one sample
    _, one_graph, _ = next(iter(data_enc.values()))
    hyperparam_dim = one_graph.hyperparams.size(1)
    
    # 4) Compute output_dim from one label
    _, _, one_label = next(iter(data_enc.values()))
    output_dim = one_label.size(-1)
    
    # 5) Set any embedding/hidden sizes you like
    type_embedding_dim = 16
    hidden_dim         = 64
    lr                 = 0.01
    epochs             = 25
    
    # 6) Run the test with computed params
    model, iou, giou, ciou = testCombinedFeatureGCN(
        num_module_types, type_embedding_dim, hyperparam_dim,
        hidden_dim, output_dim, epoch=25
    )
    print(f"CombinedFeatureGCN final test IOU Loss: {iou:.4f}")
    print(f"CombinedFeatureGCN final test GIOU Loss: {giou:.4f}")
    print(f"CombinedFeatureGCN final test CIOU Loss: {ciou:.4f}")

    # 1) Standardize hyperparams once:
    all_hp = torch.cat([g.hyperparams for _,g,_ in data_enc.values()], dim=0)
    mu, std = all_hp.mean(0), all_hp.std(0)

    # 2) Lower LR + weight decay + unified MSE:
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for vector, data, labels in train_loader_encoded:
        data.x = data.x.long()
        # normalize hyperparams
        data.hyperparams = (data.hyperparams - mu)/(std+1e-6)

        optimizer.zero_grad()
        outputs = model(data)
        loss    = criterion(outputs, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()