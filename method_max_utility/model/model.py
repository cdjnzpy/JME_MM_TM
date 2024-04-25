import torch.nn as nn
import torch

class NN_net_phi(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.activation = nn.ReLU()

        self.Linear1 = nn.Linear(args.input_dim, args.hidden_layer[0])
        self.Linear2 = nn.Linear(args.hidden_layer[0], args.hidden_layer[1])
        self.Linear3 = nn.Linear(args.hidden_layer[1], args.hidden_layer[2])
        self.Linear_last = nn.Linear(args.hidden_layer[2],args.output_dim)

    def forward(self, x):
        #正则化维度
        try:
            shocks = x[:,0].unsqueeze(1)
            w_grids = x[:,1].unsqueeze(1)            

            shock_norm = shocks/self.args.sigma
            w_grids_norm = (w_grids-self.args.W_range[0])/(self.args.W_range[1]-self.args.W_range[0])*2.0 - 1.0

            x = torch.cat([shock_norm, w_grids_norm], dim = 1)

        except IndexError:
            shocks = x[0]
            w_grids = x[1]

            shock_norm = shocks/self.args.sigma
            w_grids_norm = (w_grids-self.args.W_range[0])/(self.args.W_range[1]-self.args.W_range[0])*2.0 - 1.0

            x = torch.tensor([shock_norm, w_grids_norm])


        x = self.Linear1(x)
        x = self.activation(x)
        #x = self.dropout(x)

        x = self.Linear2(x)
        x = self.activation(x)
        #x = self.dropout(x)

        x = self.Linear3(x)
        x = self.activation(x)
        #x = self.dropout(x)

        x = self.Linear_last(x)

        return x

