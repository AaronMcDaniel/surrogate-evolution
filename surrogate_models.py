import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torchvision.ops as ops

# mlp surrogate model
class MLP(nn.Module):
    def __init__(
            self, 
            input_size=1021, 
            hidden_sizes=[512, 256, 12], 
            activation_layer=nn.ReLU, 
            norm_layer=nn.BatchNorm1d, 
            bias=True, inplace=None, 
            dropout=0.0
    ):
        super(MLP, self).__init__()
        self.activation_layer = activation_layer
        params = {} if inplace is None else {"inplace": inplace}
        layers = []

        # build intermediate hidden layers, but not output layer
        in_dim = input_size
        for hidden_dim in hidden_sizes[:-1]:
            # linear -> norm -> activation -> dropout
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(nn.Dropout(dropout, **params))
            # output size of hidden layer is input size of next hidden layer
            in_dim = hidden_dim

        # build output layer without normalization and activation
        layers.append(nn.Linear(in_dim, hidden_sizes[-1], bias=bias))
        layers.append(nn.Dropout(dropout, **params))
        self.mlp = nn.Sequential(*layers)
        # initialize weights
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.mlp(x)
        return y
    
    def _init_weights(self, module):
        # initialize linear layer weights based on activation function
        if isinstance(module, nn.Linear):
            # if relu, use he norm
            if self.activation_layer in [nn.ReLU, nn.LeakyReLU]:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            # if sigmoid or tanh, use xavier norm
            elif self.activation_layer in [nn.Sigmoid, nn.Tanh]:
                nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

def plot_model_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.hist(param.data.cpu().numpy().flatten(), bins=100)
            plt.title(f'Weight Distribution for {name}')
            plt.xlabel('Weight values')
            plt.ylabel('Frequency')
            plt.savefig(f'{name}_weight_distribution.png')

# model = MLP()
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
# data = torch.randn((16, 976), dtype=torch.float32, device=device)
# label = torch.randn((16, 12), dtype=torch.float32, device=device)
# model.train()
# output = model(data)
# train_criterion = nn.MSELoss()
# loss = train_criterion(output, label)
# print(loss)
# val_criterion = nn.MSELoss(reduction='none')
# # (16, 12) matrix of 12 mse losses for each image in batch of 16
# loss_matrix = val_criterion(output, label)
# # meaned losses per metric
# loss_means = torch.mean(loss_matrix, dim=0)
# print(loss_means)