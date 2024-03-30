import os

import numpy as np
from sklearn.model_selection import train_test_split

from .. import Configurator
from nnTrainer.data.Database import DatabaseLoader

class Guard():
    def __init__(self):
        self.config: Configurator = Configurator()
        
        self.train_ID = self.config.get_inputs('train_ID')

        self.features = self.config.get_json('features')
        self.targets = self.config.get_json('targets')

        self.random_state = self.config.get_custom('random_state')

        self.input_path = os.path.join('parameters', 'inputs')
        self.output_path = os.path.join('parameters', 'outputs')

        if not os.path.isdir('parameters'):
            os.makedirs(self.input_path)
            os.makedirs(self.output_path)

        self.inputs_list = os.listdir(self.input_path)
        self.outputs_list = os.listdir(self.output_path)

        self.input_file, self.output_file = self.__build_filenames()

        self.input_is_saved, self.output_is_saved = self.is_param_saved()

    def is_param_saved(self):
        inp = True
        out = True        
        
        # input
        if not os.path.isfile(self.input_file):
            inp = False

        # output
        if not os.path.isfile(self.output_file):
            out = False

        return inp, out

    def __build_filenames(self):
        input_name = f'in_{self.train_ID}_rs{self.random_state}.npy'
        output_name = f'out_{self.train_ID}_rs{self.random_state}.npy'

        input_file = os.path.join(self.input_path, input_name)
        output_file = os.path.join(self.output_path, output_name)

        return input_file, output_file

    def __select_data(self, DataFrame):
        # Select the x data
        x = np.zeros((len(DataFrame['ID']), len(self.features)))
        for i, feature in enumerate(self.features):
            x[:,i] = DataFrame[feature]

        # Select the y data
        y = np.zeros((len(DataFrame['ID']), len(self.targets)))
        for i, target in enumerate(self.targets):
            y[:, i] = DataFrame[target]

        return x, y
    
    def __compute(self, x, y):
        # inputs
        inputs = np.zeros( (2, len(self.features)) )

        for k in range(len(self.features)):
            # Mean
            inputs[0,k] = np.mean(x[:,k])
            # std
            inputs[1,k] = np.std(x[:,k])

        # outputs
        outputs = np.zeros( (2, len(self.targets)) )
        
        for l in range(len(self.targets)):
            # Min
            outputs[0,l] = np.min(y[:,l])
            # Max
            outputs[1,l] = np.max(y[:,l])

        return inputs, outputs

    def __get_train(self, x, y):
        # Split the dataset into test and train sets
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=self.random_state)

        return x_train, y_train

    def save(self, x=(0), y=(0)):
        # Verify if parameters already exist
        if self.input_is_saved and self.output_is_saved:
            return None

        # If not x and y sets given, executes DatabaseLoader
        if len(x) == 0 or len(y) == 0:
            db_loader = DatabaseLoader()

            # Load data
            DataFrame = db_loader.load_database()

            # Select data
            x, y = self.__select_data(DataFrame)

        # Get train split
        x_train, y_train = self.__get_train(x, y)

        # Compute parameters
        inputs, outputs = self.__compute(x_train, y_train)

        # save output
        np.save(self.input_file, inputs)
        
        # save output
        np.save(self.output_file, outputs)

    def load(self):
        # Load input and output file
        return np.load(self.input_file), np.load(self.output_file) 