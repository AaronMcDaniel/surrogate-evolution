import copy
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import matplotlib.pyplot as plt
import torchvision.ops as ops
import numpy as np
from scipy import linalg

# mlp surrogate model
class MLP(nn.Module):
    def __init__(
            self, 
            input_size=1021, 
            output_size=12,
            hidden_sizes=[512, 256], 
            activation_layer=nn.ReLU, 
            norm_layer=nn.BatchNorm1d, 
            bias=True, inplace=None, 
            dropout=0.3
    ):
        super(MLP, self).__init__()
        self.activation_layer = activation_layer
        params = {} if inplace is None else {"inplace": inplace}
        layers = []

        # build intermediate hidden layers, but not output layer
        in_dim = input_size
        for hidden_dim in hidden_sizes:
            # linear -> norm -> activation -> dropout
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(nn.Dropout(dropout, **params))
            # output size of hidden layer is input size of next hidden layer
            in_dim = hidden_dim

        # build output layer
        layers.append(nn.Linear(in_dim, output_size, bias=bias))
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

def svd_lstsq(AA, BB, tol=1e-5):
    U, S, Vh = torch.linalg.svd(AA, full_matrices=False)
    Spinv = torch.zeros_like(S)
    Spinv[S>tol] = 1/S[S>tol]
    #print('HERE HERE HERE', 1/S[S>tol])
    UhBB = U.adjoint() @ BB
    if Spinv.ndim!=UhBB.ndim:
      Spinv = Spinv.unsqueeze(-1)
    SpinvUhBB = Spinv * UhBB
    # if torch.isnan(Vh.adjoint() @ SpinvUhBB).any():
    #     if torch.isnan(AA).any():
    #         print('AA')
    #     if torch.isnan(BB).any():
    #         print('BB')
    #     if torch.isnan(U).any():
    #         print('U')
    #     if torch.isnan(S).any():
    #         print('S')
    #     if torch.isnan(Vh).any():
    #         print('Vh')
    #     if torch.isnan(Spinv).any():
    #         print('Spinv')
    #     if torch.isnan(UhBB).any():
    #         print('UhBB')
    #     if torch.isnan(SpinvUhBB).any():
    #         print('SpinvUhBB')
    return Vh.adjoint() @ SpinvUhBB

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        print(out_features, in_features, grid_size, spline_order)
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.spline_weight_copy = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        # if torch.isnan(self.spline_weight).any():
        #     print('uh oh')
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            # if torch.isnan(
            #     (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
            #     * self.curve2coeff(
            #         self.grid.T[self.spline_order : -self.spline_order],
            #         noise,)).any():
            #     print('copying in nan values :(')
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor, regularization: float=1e-6):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1).float()  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1).float()  # (in_features, batch_size, out_features)

        # apply ridge regularization to A
        # breakpoint()
        # I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
        # I = I.unsqueeze(0).unsqueeze(0)
        # A_reg = A.permute(0, 2, 1) + regularization * I

        solution = torch.linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        if torch.equal(self.spline_weight, self.spline_weight_copy):
            print('IT CHANGES')
        # if torch.isnan(self.spline_weight).any():
        #     print('spline_weight', self.spline_weight)
        # if torch.isnan(self.spline_scaler.unsqueeze(-1)).any():
        #     print('spline scaler')
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        #print('INITIAL X', x.shape)
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        # if torch.isnan(self.spline_weight).any():
        #     print('spline_weight')
        # if torch.equal(self.spline_weight, self.spline_weight_copy):
        #     testtesttest=0
        # else:
        #    print('IT CHANGES')
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        #print('size check', splines.shape, orig_coeff.shape)
        # if torch.isnan(splines).any():
        #     print('splines')
        # if torch.isnan(orig_coeff).any():
        #     print('orig_coeff')
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)
        # if torch.isnan(unreduced_spline_output).any():
        #     breakpoint()
        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        # if torch.isnan(unreduced_spline_output).any():
        #     print('unreduced spline output1')
        
        unreduced_spline_output = unreduced_spline_output.type(torch.float32)
        
        #print('TYPES', x.dtype, unreduced_spline_output.dtype)
        # if torch.isnan(unreduced_spline_output).any():
        #     print('unreduced spline output2')
        # if torch.isnan(self.curve2coeff(x, unreduced_spline_output)).any():
        #     if torch.isnan(unreduced_spline_output).any():
        #         print(unreduced_spline_output)
            #self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output, isnan=True))
        #else:
        #breakpoint()
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
        # if torch.isnan(self.spline_weight).any():
        #     if torch.isnan(x).any():
        #         print('x')
        #     if torch.isnan(unreduced_spline_output).any():
        #         print('unreduced_spline_output')
        #     breakpoint()
       
           

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        input_size=1021,
        output_size=12,
        hidden_sizes=[512, 256],
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):  
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        hidden_sizes = copy.deepcopy(hidden_sizes)
        hidden_sizes.append(output_size)
        hidden_sizes.insert(0, input_size)

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(hidden_sizes, hidden_sizes[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        x = torch.tanh(x)
        for layer in self.layers:
            if update_grid:
                #print('WORKS')
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    

# potential classifier model
class BinaryClassifier(nn.Module):
    def __init__(
            self, 
            input_size=1021, 
            hidden_sizes=[512, 256], 
            activation_layer=nn.ReLU, 
            norm_layer=nn.BatchNorm1d, 
            bias=True, inplace=None, 
            dropout=0.3
    ):
        super(BinaryClassifier, self).__init__()
        self.activation_layer = activation_layer
        params = {} if inplace is None else {"inplace": inplace}
        layers = []

        # Build intermediate hidden layers, but not output layer
        in_dim = input_size
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        # Build output layer for binary classification
        layers.append(nn.Linear(in_dim, 1, bias=bias))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.mlp(x)
        return y

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.activation_layer in [nn.ReLU, nn.LeakyReLU]:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif self.activation_layer in [nn.Sigmoid, nn.Tanh]:
                nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
