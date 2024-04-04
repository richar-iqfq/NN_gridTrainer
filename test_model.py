from nnTrainer.train.Models import NetHiddenLayers
import os
import torch

os.system('clear')


n_hidden_layers = 6
n_features = 10
n_targets = 4
dimension = (9, 7, 4, 9, 9, 10)
activation_functions = ('nn.Tanh()', 'nn.LeakyReLU()', 'nn.ELU()', 'nn.LeakyReLU()', 'nn.LeakyReLU()', 'nn.ReLU()', 'None')

net = NetHiddenLayers(n_features, n_targets, n_hidden_layers, dimension, activation_functions)

x = torch.tensor([
    1.0115, 2.22458, 3.1568, 4.05498, 5.12335, 6.418569, 7.1456, 8.1896, 4.55989, 5.15645
    ])

x = net.forward(x)

param = sum(p.numel() for p in net.parameters() if p.requires_grad)

print(param)

print(x)