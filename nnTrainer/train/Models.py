import torch.nn as nn

class NetHiddenLayers(nn.Module):
    '''
    Neural Network with n hidden layers

    Parameters
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    n_hidden_layers : `int`
        Number of hidden layers for model

    dimension : `tuple` `int` (n_hidden_layers elements)
        Tuple with the number of neurons in the hidden layer (n)

    activation_functions : `List` `str` (n_hidden_layers + 1 elements)
        List with the modules for each activation function in string 
        format ('nn.ReLU()', ..., n_hidden_layers + 1). If last element equals to 'None',
        not af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, n_hidden_layers, dimension, activation_functions):
        super(NetHiddenLayers, self).__init__()

        self.sequential_string = ''

        if len(dimension) != n_hidden_layers:
            raise Exception('Wrong dimension size')
        
        if len(activation_functions) != n_hidden_layers + 1:
            raise Exception('Wrong activation functions size')

        # Build linear layers
        for layer in range(n_hidden_layers):
            # First hidden layer
            if layer == 0:
                self.sequential_string += f'nn.Linear({n_features}, {dimension[layer]}), '
                self.sequential_string += f'{activation_functions[layer]}, ' if activation_functions[layer] != 'None' else ''

                # Final hidden layer
                if n_hidden_layers == 1:
                    self.sequential_string += f'nn.Linear({dimension[layer]}, {n_targets}), '
                    self.sequential_string += f'{activation_functions[layer+1]}, ' if activation_functions[layer+1] != 'None' else ''
            
            # Other hidden layers
            else:
                last_n_neurons = dimension[layer-1]

                self.sequential_string += f'nn.Linear({last_n_neurons}, {dimension[layer]}), '
                self.sequential_string += f'{activation_functions[layer]}, ' if activation_functions[layer] != 'None' else ''

                # Final hidden layer
                if layer == n_hidden_layers-1:
                    self.sequential_string += f'nn.Linear({dimension[layer]}, {n_targets}), '
                    self.sequential_string += f'{activation_functions[layer+1]}, ' if activation_functions[layer+1] != 'None' else ''

        self.model = eval(f'nn.Sequential({self.sequential_string})')

    def forward(self, x):
        return self.model(x)