import numpy as np
import os
from sklearn.model_selection import train_test_split

from modules.DatabaseLoader import DatabaseLoader
from modules.ParameterGuard import ParameterGuard

class PreprocessData():
    def __init__(self, config):
        self.config = config

        self.features = config.json['features']
        self.n_features = len(self.features)

        self.targets = config.json['targets']
        self.n_targets = len(self.targets)


        self.train_ID = config.inputs['train_ID']
        self.scale_y = config.inputs['scale_y']
        
        self.random_state = config.custom['random_state']

        # Build the data structure
        self.ID, self.x, self.y = self.__build_structure()

        # Compute parameters
        guard = ParameterGuard(self.config)
        guard.save(self.x, self.y)

        # Load parameters in class
        self.in_param, self.out_param = self.__get_parameters()

        # Scale the data
        self.x_scaled, self.y_scaled = self.Scale(self.x, self.y)

        # Split data
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.Split_Data(self.x_scaled, self.y_scaled)

    def __get_parameters(self):
        # Check if specific param_file is given
        if self.config.configurations['specific_param_file']:
            input_file = os.path.join('parameters', 'inputs', f"inputs_{self.config.configurations['specific_param_file']}")
            output_file = os.path.join('parameters', 'outputs', f"outputs_{self.config.configurations['specific_param_file']}")
        else:
            input_file, output_file = self.__get_filenames()

        in_param = np.load(input_file)
        out_param = np.load(output_file)

        return in_param, out_param
    
    def __get_filenames(self):
        input_name = f'in_{self.train_ID}_rs{self.random_state}.npy'
        output_name = f'out_{self.train_ID}_rs{self.random_state}.npy'

        input_file = os.path.join('parameters', 'inputs', input_name)
        output_file = os.path.join('parameters', 'outputs', output_name)

        return input_file, output_file

    def __build_structure(self):
        # Load DataFrame with data
        Loader = DatabaseLoader(self.config)
        DataFrame = Loader.load_database()

        # Define ID values
        ID = DataFrame['ID']

        # Select the x data
        x = np.zeros((len(DataFrame['ID']), self.n_features))
        for i, feature in enumerate(self.features):
            x[:,i] = DataFrame[feature]

        # Select the y data
        y = np.zeros((len(DataFrame['ID']), self.n_targets))
        for i, target in enumerate(self.targets):
            y[:, i] = DataFrame[target]

        return ID, x, y

    def x_scale_routine(self, x):
        mean = self.in_param[0] # mean
        std = self.in_param[1] # std

        x_scaled = np.zeros_like(x)

        for i, sample in enumerate(x):
            for j, value in enumerate(sample):
                x_scaled[i, j] = ( value - mean[j] ) / std[j]

        return x_scaled
    
    def x_unscale_routine(self, x_scaled):
        mean = self.in_param[0] # mean
        std = self.in_param[1] # std

        x = np.zeros_like(x_scaled)

        for i, sample in enumerate(x_scaled):
            for j, value in enumerate(sample):
                x[i,j] = ( value * std[j] ) + mean[j]

        return x
    
    def y_scale_routine(self, y):
        v_min = 0
        v_max = 1

        Ymin = self.out_param[0]
        Ymax = self.out_param[1]

        y_scaled = np.zeros_like(y)

        for i, y_value in enumerate(y):
            Ystd = ( y_value - Ymin ) / ( Ymax - Ymin )
            
            y_scaled[i] = ( Ystd * (v_max - v_min) ) + v_min

        return y_scaled
    
    def y_unscale_routine(self, y_scaled):
        v_min = 0
        v_max = 1

        Ymin = self.out_param[0]
        Ymax = self.out_param[1]

        y = np.zeros_like(y_scaled)

        for i, y_scaled_value in enumerate(y_scaled):
            y[i] = ( ( (y_scaled_value - v_min)/(v_max - v_min) ) * (Ymax - Ymin) ) + Ymin

        return y

    def Scale(self, x, y):
        x_scale = self.x_scale_routine(x)
        
        if self.scale_y:
            y_scale = self.y_scale_routine(y)
        else:
            y_scale = y

        return x_scale, y_scale
    
    def Unscale(self, x_scaled, y_scaled):
        x = self.x_unscale_routine(x_scaled)

        if self.scale_y:
            y = self.y_unscale_routine(y_scaled)
        else:
            y = y_scaled

        return x, y

    def Split_Data(self, x, y):
        # Split the dataset into test and train sets
        x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.2, random_state=self.random_state)

        # Split the t dataset into val and train set
        x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.25, random_state=self.random_state)

        # 80% Train dataset
        # 15% Val dataset
        # 5% Test dataset

        return x_train, x_val, x_test, y_train, y_val, y_test

    def Retrieve_Processed(self):
        return self.ID, self.x_scaled, self.y_scaled
        
    def Retrieve_Splitted(self):
        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test