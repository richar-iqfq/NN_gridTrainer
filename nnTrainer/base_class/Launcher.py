import logging
import time
import copy

from .. import Configurator
from nnTrainer.train.Trainer import Trainer
from nnTrainer.data.Sql import SqlReader

class MainLauncher():
    '''
    Customizable class to launch multiple architectures to training and watch the progress in status bar
    '''
    def __init__(self):
        # Hidden size to start the searching after the grid step
        self.initial_hidden_size_after_grid: int = 1

        # config property
        self.config: Configurator = Configurator()

        # reader
        self.reader = SqlReader()

        # Number of parameters tolerance
        self.tol: float = self.config.get_configurations('network_tolerance')

        # Architecture loading
        self.num_features: int = self.config.get_json('num_features')
        self.num_targets: int = self.config.get_json('num_targets')
        self.optimizers: list = self.config.get_json('optimizers')
        self.loss_functions: list = self.config.get_json('loss_functions')
        self.af_list: tuple = copy.deepcopy(self.config.get_json('af_valid'))
        self.af_list.remove('None')

        # Inputs
        self.parted: int = self.config.get_custom('parted')
        self.extra_name: str = self.config.get_custom('extra_filename')
        self.seed: int = self.config.get_custom('seed')
        self.lineal_output: bool = self.config.get_custom('lineal_output')

        # Configurations
        self.max_hidden_layers: int = self.config.get_configurations('max_hidden_layers')
        self.min_neurons: int = self.config.get_configurations('min_neurons')
        self.max_neurons: int = self.config.get_configurations('max_neurons')
        self.lr_range: float = self.config.get_configurations('learning_rate_range')
        self.bz_range: tuple = self.config.get_configurations('batch_size_range')
        self.tries: int = self.config.get_configurations('n_tries')
        self.n_networks: int = self.config.get_configurations('n_networks')
        self.start_point: int = self.config.get_configurations('start_point')
        self.save_plots: bool = self.config.get_configurations('save_plots')
        self.reader_criteria: str = self.config.get_configurations('reader_criteria')
        self.workers: int = self.config.get_configurations('workers')
        self.train_ID: str = self.config.get_inputs('train_ID')

        logging.basicConfig(filename=f'logs/{self.train_ID}.log', level=logging.INFO)

    def build_network_name(self, hidden_size: int):
        return f'Net_{hidden_size}Hlayer'

    def launch(self, trainer: Trainer) -> tuple:
        '''
        launch the training and assure the number of parameters is on range

        Parameters
        ----------
        trainer `object of class Trainer`
        '''
        monitoring = self.config.get_monitoring('monitoring_state')
        n_parameters = trainer.parameters_count()
        n_train = trainer.database_size()
        message = ''
        flag = True
        
        if self.tol <= 0.12:
            if n_parameters <= self.tol*n_train:
                time.sleep(1)
                flag, message = trainer.start_training(save_plots=self.save_plots, monitoring=monitoring, allow_print=False)

                trainer.close_plots()
                trainer.reset()
        
        else:
            if n_parameters > 0.12*n_train and n_parameters <= self.tol*n_train:
                time.sleep(1)
                flag, message = trainer.start_training(save_plots=self.save_plots, monitoring=monitoring)

                trainer.close_plots()
                trainer.reset()

        if message != '':
            logging.info(message)

        return flag, trainer

    def recover_network(self, hidden_layers: int, step: str, worst: bool=False) -> list:
        launch_code = self.config.get_inputs('train_ID')
        criteria = self.config.get_configurations('reader_criteria')
        n_networks = self.config.get_configurations('n_networks')
        
        # Load better network parameters
        try:
            better_network = self.reader.recover_best(launch_code, hidden_layers, step, criteria, n_values=n_networks, worst=worst)
        except:
            better_network = None
        
        return better_network
    
    def run(self, previous_step: str, network: dict=False):
        # Empty method, implemented by step
        pass