from nnTrainer.launch.Main import MainLauncher
from nnTrainer.train.Trainer import Trainer

class Testing(MainLauncher):
    def __init__(self):
        super().__init__()

        self.actual_step = 'Testing'

    def run(self, network: dict, overview: bool=True, monitoring: bool=False) -> None:
        '''
        Run a specific model and save it to folder Models/recovering

        Parameters
        ----------
        network : object of class Reader().recover_best()
            Dictionary with the parameters for the model to save
        
        overview : `bool`
            If True will print on console the full network information

        monitoring : `bool`
            If True the file will keep the better result obtained during training
        '''
        # Set random state from configurator class
        self.config.update(
            random_state = network['random_state']
        )

        # Set architecture and hyperparameters
        hidden_layers = network['hidden_layers']
        Network = self.Networks[hidden_layers]

        architecture = {
            'model' : Network,
            'num_targets' : self.num_targets,
            'num_features' : self.config.get_json('num_features'),
            'dimension' : network['dimension'],
            'activation_functions' : network['activation_functions'],
            'optimizer' : network['optimizer'],
            'criterion' : network['criterion']
        }

        Hyperparameters = {
            'num_epochs' : self.config.get_hyperparameters('num_epochs'),
            'batch_size' : int(network['batch_size']),
            'learning_rate' : round(network['lr'], 9)
        }

        extra_name = self.config.get_custom('extra_filename')
        save_plots = self.config.get_configurations('save_plots')

        file = Network + f'{extra_name}.csv'
        tr = Trainer(file, architecture, Hyperparameters, step='recovering')
        
        if overview:
            tr.overview()

        tr.start_training(write=True , allow_print=True, save_plots=save_plots, monitoring=monitoring)