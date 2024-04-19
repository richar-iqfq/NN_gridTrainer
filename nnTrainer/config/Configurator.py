import os
import json
import copy
from configparser import ConfigParser
from nnTrainer.base_class.Singleton import SingletonMeta

# Configurator class
class Configurator(metaclass=SingletonMeta):
    '''
    Configurator class that contains all the information needed to run the trainings
    '''
    def __init__(self):
        # Main
        self.database_path: str = ''

        # Configurations
        self.configurations: dict = {
            'max_hidden_layers' : 5, # Max number of hidden layers
            'min_neurons' : 5, # Min number of neurons on each layer
            'max_neurons' : 20,  # Max number of neurons on each layer
            'learning_rate_range' : (0.001 , 0.1), # Range to choose the lr values randomly
            'batch_size_range' : (2, 1000), # Range to choose the batch size randomly
            'n_tries' : 600, # Number of tries for alleatory search
            'n_networks' : 1, # Number of selected networks from grid step to train in lr_tuning and after.
            'start_point' : 1, # Number of the initial hidden layer
            'save_plots' : True, # If true, the trainer will storage all the metric plots
            'save_full_predictions' : False, # If true, generates the final file with the targets unscaled
            'workers' : 0, # Number of workers for the training
            'reader_criteria' : 'R2Test_i', # criteria to choose the better network
            'percent_outliers' : 0.08, # Tolerance for outliers
            'network_tolerance' : 0.12, # Tolerance for n_parameters percentage
            'drop' : False, # File name with the extra molecules to be dropped from database
            'config_file' : 'config.json', # Configuration file with features and other relevant parameters
            'specific_param_file' : False, # Especific parameters filename for scaling
            'outliers_strategy' : 'statistical_difference', # The outliers definition strategy to be used
            'training_codes_database' : 'Training_results/dabaseCodes.db', # Database to store training codes
        }
        
        # Hyperparameters
        self.hyperparameters: dict = {
            'num_epochs' : 600, # Number of epochs
            'batch_size' : 'All', # Size of batch if 'All', the batch size is the length of dataset
            'learning_rate' : 0.001 # Learning Rate
        }

        # Options
        self.loss: dict = {
            'optimizer' : 'AdamW', # Optimizer
            'criterion' : 'nn.L1Loss()' # Loss function
        }

        # Target
        self.inputs: dict = {
            'database' : 'dataset_final_sorted_3.1.0.csv',
            'v_min' : [0, 0, 0, 0, 0, 0], # Min value to scale each target value
            'v_max' : [1, 1, 1, 1, 1, 1], # Max value to scale each target value
            'drop_file' : None, # File with the list of molecules to be drop from database in training
            'train_ID' : 'T000' # General ID for the training
        }

        # Custom
        self.custom: dict = {
            'extra_filename' : 'default', # Extra words that'll join the output file name
            'lineal_output' : False, # If true, no output activation layer will be used
            'seed' : 3358, # Seed for the alleatory spliting
            'random_state' : 123, # Random state for the Split_dataset function
            'parted' : None, # Can be 1, 2 or 3. Refers to the split of the training.
        }

        # monitor
        self.monitoring: dict = {
            'best_acc' : 0,
            'best_epoch' : 0
        }

        # Cuda
        self.cuda: dict = {
            'limit_threads' : False # If true, training will be limited to use only one cpu core
        }

        # Register Components
        self.components: list = [self.configurations,
                                 self.hyperparameters,
                                 self.loss,
                                 self.custom,
                                 self.inputs,
                                 self.cuda,
                                 self.monitoring]
        
        # Valid Keys
        self.valid_keys: list = []

        for component in self.components:
            for key in component:
                self.valid_keys.append(key)
        
        # json
        self.json:dict = {
            'features': '',
            'num_features': '',
            'targets': '',
            'num_targets': '',
            'optimizers': '',
            'loss_functions': '',
            'af_valid': ''
        }

        # config.ini
        self.config_object: ConfigParser = ConfigParser()
        self.config_object.optionxform = str

    def update(self, **kwargs: dict) -> None:
        '''
        Udate the default configuration values

        Parameters
        ----------
        **kwargs:
            Valid keys: max_hidden_layers, min_neurons, max_neurons, learning_rate_range,
            batch_rate_range, n_tries, n_networks, start_point, save_plots, workers,
            save_full_predictions, reader_criteria, percent_outliers, n_pics, drop,
            drop_model_outliers, config_file, specific_param_file, num_epochs, batch_size,
            learning_rate, optimizer, criterion, target, b, alpha, drop_file,
            train_ID, lineal_output, extra_filename, parted, seed, random_state,
            limit_threads
        '''
        self.__validateParams(kwargs)

        for key in kwargs:
            for component in self.components:
                if key in component:
                    component[key] = kwargs[key]

        # If not given, select config_file by train_ID
        if self.configurations['config_file'] == 'config.json':
            self.configurations['config_file'] = f"config{self.inputs['train_ID']}.json"

        # If not given, build extra_filename by b and train_ID
        if self.custom['extra_filename'] == 'default':
            self.custom['extra_filename'] = f"{self.inputs['train_ID']}"

        self.__build_database_routes()
        self.__import_json_values()

    def __str__(self) -> str:
        text_chain = '-'*20
        for component in self.components:
            for key in component:
                text_chain += f'\n{key} -> {component[key]}'
            text_chain += '\n' + '-'*20

        text_chain += f'\ndatabase -> {self.database_path}'

        return text_chain

    def validParameters(self) -> None:
        '''
        Show allowed keys
        '''
        print(f'Valid parameters:\n{str(self.valid_keys)}')

    def __validateParams(self, values: dict) -> None:
        '''
        Validate the incoming parameters before updating
        '''
        for key in values:
            if key not in self.valid_keys:
                message = f'Unknown value -> {key}'
                raise Exception(message)
            
    def __build_database_routes(self) -> None:
        '''
        Build database routes by checking the existence of data files
        '''
        database_path = os.path.join('dataset', self.inputs['database'])
        if not os.path.isfile(database_path):
            message = f'{database_path} not found, please check'
            raise Exception(message)

        self.database_path = database_path

        if self.configurations['drop']:
            drop_file = os.path.join('dataset', 'drop_molecules', self.configurations['drop'])
            self.inputs['drop_file'] = drop_file

            if not os.path.isfile(drop_file):
                message = f"drop_file -> {drop_file} doesn't exists"
                raise Exception(message)

    def __import_json_values(self) -> None:
        '''
        Import configuration data from config.json file
        '''
        config_file = os.path.join('config', self.configurations['config_file'])

        if not os.path.isfile(config_file):
            message = f"Configuration file -> {config_file} don't exists!"
            raise Exception(message)

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Rewrite values
        self.json['features'] = config['data_structure']['features']
        self.json['num_features'] = len(self.json['features'])

        self.json['targets'] = config['data_structure']['targets']
        self.json['num_targets'] = len(self.json['targets'])
        
        self.json['optimizers'] = config['network_structure']['optimizers']
        self.json['loss_functions'] = config['network_structure']['loss_functions']
        self.json['af_valid'] = config['network_structure']['af_valid']

    def save_ini(self, config_file: str) -> None:
        '''
        Save data to config_file.ini.
        '''
        self.config_object['configurations'] = {
            'drop' : self.configurations['drop'],
            'config_file' : self.configurations['config_file'],
            'specific_param_file' : self.configurations['specific_param_file']
        }

        self.config_object['inputs'] = {
            'database' : self.inputs['database'],
            'train_ID' : self.inputs['train_ID']
        }

        self.config_object['custom'] = {
            'seed' : self.custom['seed'],
            'lineal_output' : self.custom['lineal_output'],
        }

        self.config_object['monitoring'] = {
            'best_mean_acc' : self.monitoring['best_acc'],
            'best_epoch' : self.monitoring['best_epoch']
        }

        # Write inside config.ini file
        with open(config_file, 'w') as conf:
            self.config_object.write(conf)

    def load_ini(self, config_path: str) -> None:
        '''
        Load data from config_file.ini.
        '''
        self.config_object.read(config_path)

        # Load vales
        ini_configurations = self.config_object['configurations']
        ini_inputs = self.config_object['inputs']
        ini_custom = self.config_object['custom']

        # Update class
        for key in ini_configurations:
            self.configurations[key] = ini_configurations[key]

        for key in ini_inputs:
            self.inputs[key] = ini_inputs[key]

        for key in ini_custom:
            self.custom[key] = ini_custom[key]

    def get_target(self):
        return copy.deepcopy(self.target)

    def get_database(self):
        return copy.deepcopy(self.database)

    def get_smiles_database(self):
        return copy.deepcopy(self.smiles_database)

    def get_configurations(self, key):
        return self.configurations[key]
    
    def get_hyperparameters(self, key=None):
        if key:
            return self.hyperparameters[key]
        else:
            return copy.deepcopy(self.hyperparameters)
    
    def get_loss(self, key):
        return self.loss[key]
    
    def get_inputs(self, key):
        return self.inputs[key]
    
    def get_custom(self, key):
        return self.custom[key]
    
    def get_monitoring(self, key):
        return self.monitoring[key]
    
    def get_cuda(self, key):
        return self.cuda[key]
    
    def get_valid_keys(self):
        return copy.deepcopy(self.valid_keys)
    
    def get_json(self, key):
        return self.json[key]