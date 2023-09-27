import os
import json
from configparser import ConfigParser

class Configurator():
    def __init__(self):
        # Main
        self.database_path = ''

        # Configurations
        self.configurations = {
            'max_hidden_layers' : 5, # Max number of hidden layers
            'min_neurons' : 5, # Min number of neurons on each layer
            'max_neurons' : 20,  # Max number of neurons on each layer
            'learning_rate_range' : (0.001 , 0.1), # Range to choose the lr values randomly
            'batch_size_range' : (30, 1500), # Range to choose the batch size randomly
            'n_tries' : 600, # Number of tries for alleatory search
            'start_point' : 1, # Number of the initial hidden layer
            'save_plots' : True, # If true, the trainer will storage all the metric plots
            'save_full_predictions' : False, # If true, generates the final file with the targets unscaled
            'workers' : 0, # Number of workers for the training
            'reader_criteria' : 'acc_val', # criteria to choose the better network
            'percent_outliers' : 0.08, # Tolerance for outliers
            'n_pics' : 30, # Number of molecules to build pictures
            'drop_model_outliers' : False, # If True, will extract the outliers obtained in full training
            'drop' : False, # File name with the extra molecules to be dropped from database
            'config_file' : 'default.json', # Configuration file with features and other relevant parameters
            'specific_param_file' : False # Especific parameters filename for scaling
        }
        
        # Hyperparameters
        self.hyperparameters = {
            'num_epochs' : 600, # Number of epochs
            'batch_size' : 'All', # Size of batch if 'All', the batch size is the length of dataset
            'learning_rate' : 0.001 # Learning Rate
        }

        # Options
        self.loss = {
            'optimizer' : 'AdamW', # Optimizer
            'criterion' : 'nn.L1Loss()' # Loss function
        }

        # Target
        self.inputs = {
            'database' : 'dataset_final_sorted_2.4.3.csv', # Csv database name (not path)
            'scale_y' : True, # If true, target will be scaled to the interval (0, 1)
            'drop_file' : None, # File with the list of molecules to be drop from database in training
            'train_ID' : 'A000' # General ID for the training
        }

        # Custom
        self.custom = {
            'lineal_output' : False, # If true, no output activation layer will be used
            'extra_filename' : 'default', # Extra words that'll join the output file name
            'parted' : None, # Can be 1, 2 or 3. Refers to the split of the training.
            'seed' : 3358, # Seed for the alleatory spliting
            'random_state' : 123 # Random state for the Split_dataset function
        }

        # monitor
        self.monitoring = {
            'best_acc' : 0,
            'best_epoch' : 0
        }

        self.cuda = {
            'limit_threads' : False # If true, training will be limited to use only one cpu core
        }

        # Components
        self.components = [self.configurations,
                           self.hyperparameters,
                           self.loss,
                           self.custom,
                           self.inputs,
                           self.cuda,
                           self.monitoring]
        
        # Valid Keys
        self.valid_keys = []

        for component in self.components:
            for key in component:
                self.valid_keys.append(key)
        
        # json
        self.json = {}

        # config.ini
        self.config_object = ConfigParser()
        self.config_object.optionxform = str

    def update(self, **kwargs):
        self.__validateParams(kwargs)

        for key in kwargs:
            for component in self.components:
                if key in component:
                    component[key] = kwargs[key]

        # If not given, select config_file by train_ID
        if self.configurations['config_file'] == 'default.json':
            self.configurations['config_file'] = f"config{self.inputs['train_ID']}.json"

        # If not given, build extra_filename by train_ID
        if self.custom['extra_filename'] == 'default':
            self.custom['extra_filename'] = f"{self.inputs['train_ID']}"

        self.__build_database_routes()
        self.__import_json_values()

    def __str__(self):
        text_chain = '-'*20
        for component in self.components:
            for key in component:
                text_chain += f'\n{key} -> {component[key]}'
            text_chain += '\n' + '-'*20

        text_chain += f'\ndatabase -> {self.database_path}'

        return text_chain

    def validParameters(self):
        print(f'Valid parameters:\n{str(self.valid_keys)}')

    def __validateParams(self, values):
        for key in values:
            if key not in self.valid_keys:
                message = f'Unknown value -> {key}'
                raise Exception(message)
            
    def __build_database_routes(self):
        database_path = os.path.join('dataset', self.inputs['database'])
        if not os.path.isfile(database_path):
            message = f'{database_path} not found, please check b and alpha'
            raise Exception(message)

        self.database_path = database_path

        if self.configurations['drop']:
            drop_file = os.path.join('dataset', 'drop_molecules', self.configurations['drop'])
            self.inputs['drop_file'] = drop_file

            if not os.path.isfile(drop_file):
                message = f"drop_file -> {drop_file} don't exists"
                raise Exception(message)

    def __import_json_values(self):
        config_file = os.path.join('config', self.configurations['config_file'])

        if not os.path.isfile(config_file):
            message = f"Configuration file -> {config_file} don't exists!"
            raise Exception(message)

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Assign values
        self.json['features'] = config['data_structure']['features']
        self.json['num_features'] = len(self.json['features'])

        self.json['targets'] = config['data_structure']['targets']
        self.json['num_targets'] = len(self.json['targets'])

        self.json['optimizers'] = config['network_structure']['optimizers']
        self.json['loss_functions'] = config['network_structure']['loss_functions']
        self.json['af_valid'] = config['network_structure']['af_valid']

    def save_ini(self, config_file):
        self.config_object['configurations'] = {
            'drop' : self.configurations['drop'],
            'config_file' : self.configurations['config_file'],
            'specific_param_file' : self.configurations['specific_param_file']
        }

        self.config_object['inputs'] = {
            'database' : self.inputs['database'],
            'scale_y' : self.inputs['scale_y'],
            'train_ID' : self.inputs['train_ID']
        }

        self.config_object['custom'] = {
            'lineal_output' : self.custom['lineal_output'],
            'seed' : self.custom['seed']
        }

        self.config_object['monitoring'] = {
            'best_acc' : self.monitoring['best_acc'],
            'best_epoch' : self.monitoring['best_epoch']
        }

        # Write inside config.ini file
        with open(config_file, 'w') as conf:
            self.config_object.write(conf)

    def load_ini(self, config_path):
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

    def status(self):
        
        info = [
            f'Config : \n{self.configurations}\n',
            f'Hyperparameters : \n{self.hyperparameters}\n',
            f'Loss : \n{self.loss}\n',
            f'Inputs : \n{self.inputs}\n',
            f'Custom : \n{self.custom}\n',
            f'Monitoring : \n{self.monitoring}'
            f'Json : \n{self.json}\n'
        ]
        
        for line in info:
            print(line)