from modules.Models import Net_1Hlayer
from modules.Configurator import Configurator
from modules.PreprocessData import PreprocessData
import torch
import torch.nn as nn
from modules.Trainer import Trainer
import os

os.system('clear')

def Test_model(config):
    # Set architecture and hyperparameters
    Network = 'Net_4Hlayer'

    architecture = {
        'model' : Network,
        'num_features' : config.json['num_features'],
        'dimension' : (17, 17, 16, 20),
        'activation_functions' : ('nn.ELU()', 'nn.LeakyReLU()', 'nn.Sigmoid()', 'nn.ReLU()', 'nn.ReLU()'),
        'optimizer' : 'Adam',
        'criterion' : 'nn.L1Loss()',
        'num_targets' : config.json['num_targets']
    }

    Hyperparameters = {
        'num_epochs' : 50,
        'batch_size' : 256,
        'learning_rate' : 0.001
    }

    extra_name = config.custom['extra_filename']
    save_plots = config.configurations['save_plots']

    file = Network + f'{extra_name}.csv'
    tr = Trainer(file, architecture, Hyperparameters, config, mode='complete', extra_route='grid')
    
    tr.overview()

    tr.start_training(save_plots=False, allow_print=True, monitoring=False)

config = Configurator()
config.update(
    config_file='configA000.json'
)

Test_model(config)