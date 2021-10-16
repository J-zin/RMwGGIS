import torch
import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "swish": Swish(),
    "square": Lambda(lambda x: x**2),
    "identity": Lambda(lambda x: x),
}

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity='elu', act_last=None, bn=False, dropout=-1):
        super(MLP, self).__init__()
        self.act_last = act_last
        self.nonlinearity = nonlinearity
        self.input_dim = input_dim
        self.bn = bn

        if isinstance(hidden_dims, str):
            hidden_dims = list(map(int, hidden_dims.split("-")))
        assert len(hidden_dims)
        hidden_dims = [input_dim] + hidden_dims
        self.output_size = hidden_dims[-1]

        list_layers = []

        for i in range(1, len(hidden_dims)):
            list_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            if i + 1 < len(hidden_dims):  # not the last layer
                if self.bn:
                    bnorm_layer = nn.BatchNorm1d(hidden_dims[i])
                    list_layers.append(bnorm_layer)
                list_layers.append(NONLINEARITIES[self.nonlinearity])
                if dropout > 0:
                    list_layers.append(nn.Dropout(dropout))
            else:
                if act_last is not None:
                    list_layers.append(NONLINEARITIES[act_last])

        self.main = nn.Sequential(*list_layers)

    def forward(self, z):
        x = self.main(z)
        return x

class MLPScore(nn.Module):
    def __init__(self, input_dim, hidden_dims, scale=1.0, nonlinearity='swish', act_last=None, bn=False, dropout=-1, bound=-1):
        super(MLPScore, self).__init__()
        self.scale = scale
        self.bound = bound
        self.mlp = MLP(input_dim, hidden_dims, nonlinearity, act_last, bn, dropout)

    def forward(self, z):
        raw_score = self.mlp(z.float() / self.scale)
        if self.bound > 0:
            raw_score = torch.clamp(raw_score, min=-self.bound, max=self.bound)
        return raw_score
