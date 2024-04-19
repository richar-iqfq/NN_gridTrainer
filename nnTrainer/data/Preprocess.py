import os

import numpy as np
from sklearn.model_selection import train_test_split

from .. import Configurator
from nnTrainer.data.Database import DatabaseLoader
from nnTrainer.data.Parameters import Guard

class PreprocessData():
    def __init__(self):
        '''
        Preprocess data, scale, unscale and assures if parameters are saved on
        disk

        Parameters
        ----------
        config object of `class` Configurator
        '''
        self.config: Configurator = Configurator()

        self.features = self.config.get_json('features')
        self.n_features = len(self.features)

        self.targets = self.config.get_json('targets')
        self.n_targets = len(self.targets)

        self.v_min = self.config.get_inputs('v_min')
        self.v_max = self.config.get_inputs('v_max')

        self.train_ID = self.config.get_inputs('train_ID')
        self.lineal_output = self.config.get_custom('lineal_output')
        self.scale_y = False if self.lineal_output else True
        
        self.random_state = self.config.get_custom('random_state')

        # Build the data structure
        self.ID, self.x, self.y = self.__build_structure()

        # Compute parameters
        self.parameter_guard = Guard()
        self.parameter_guard.save(self.x, self.y)

        # Load parameters in class
        self.in_param, self.out_param = self.parameter_guard.load()

        # Scale the data
        self.x_scaled, self.y_scaled = self.Scale(self.x, self.y)

        # Split data
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.Split_Data(self.x_scaled, self.y_scaled)

    def __build_structure(self):
        '''
        Build the data structure, ID, x and y arrays.
        '''
        # Load DataFrame with data
        Loader = DatabaseLoader()
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
        '''
        Scaling routine for features (x), use the mean and standard deviation
        of the training splitted dataset.
        '''
        mean = self.in_param[0] # mean
        std = self.in_param[1] # std

        x_scaled = np.zeros_like(x)

        for i, sample in enumerate(x):
            for j, value in enumerate(sample):
                x_scaled[i, j] = ( value - mean[j] ) / std[j]

        return x_scaled
    
    def x_unscale_routine(self, x_scaled):
        '''
        Unscaling routine for features (x), use the mean and standard deviation
        of the training splitted dataset.
        '''
        mean = self.in_param[0] # mean
        std = self.in_param[1] # std

        x = np.zeros_like(x_scaled)

        for i, sample in enumerate(x_scaled):
            for j, value in enumerate(sample):
                x[i,j] = ( value * std[j] ) + mean[j]

        return x
    
    def y_scale_routine(self, y):
        '''
        Scaling routine for targets (y), use the min and max value
        of the training splitted dataset.
        '''
        Ymin = self.out_param[0]
        Ymax = self.out_param[1]

        y_scaled = np.zeros_like(y)

        for i, sample in enumerate(y):
            for j, value in enumerate(sample):
                Ystd = ( value - Ymin[j] ) / ( Ymax[j] - Ymin[j] )
                y_scaled[i,j] = ( Ystd * (self.v_max[j] - self.v_min[j]) ) + self.v_min[j]

        return y_scaled
    
    def y_unscale_routine(self, y_scaled):
        '''
        Unscaling routine for targets (y), use the min and max value
        of the training splitted dataset.
        '''
        Ymin = self.out_param[0]
        Ymax = self.out_param[1]

        y = np.zeros_like(y_scaled)
        
        for i, sample in enumerate(y_scaled):
            for j, value in enumerate(sample):
                y[i,j] = ( ( (value - self.v_min[j])/(self.v_max[j] - self.v_min[j]) ) * (Ymax[j] - Ymin[j]) ) + Ymin[j]

        return y

    def Scale(self, x, y):
        '''
        Compute the scaling of x and y if requested.
        '''
        x_scale = self.x_scale_routine(x)
        
        if self.scale_y:
            y_scale = self.y_scale_routine(y)
        else:
            y_scale = y

        return x_scale, y_scale
    
    def Unscale(self, x_scaled, y_scaled):
        '''
        Compute the unscaling of x and y if requested.
        '''
        x = self.x_unscale_routine(x_scaled)

        if self.scale_y:
            y = self.y_unscale_routine(y_scaled)
        else:
            y = y_scaled

        return x, y

    def Split_Data(self, x, y):
        '''
        Split data into training, validation y testing set.
        '''
        # Split the dataset into test and train sets
        x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.2, random_state=self.random_state)

        # Split the t dataset into val and train set
        x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.25, random_state=self.random_state)

        # 80% Train dataset
        # 15% Val dataset
        # 5% Test dataset

        return x_train, x_val, x_test, y_train, y_val, y_test

    def Retrieve_Processed(self):
        '''
        Return the scaled values.
        '''
        return self.ID, self.x_scaled, self.y_scaled
        
    def Retrieve_Splitted(self):
        '''
        Return the splitted data into training, validation and testing sets.
        '''
        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test