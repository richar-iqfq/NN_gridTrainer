import numpy as np
import pandas as pd

class DatabaseLoader():
    def __init__(self, config):
        # Variable assingment
        self.features = config.json['features']
        self.targets = config.json['targets']

        self.DataFrame = pd.read_csv(config.database_path)

        self.initial_size = len(self.DataFrame)
        self.after_unconverged_size = 0
        self.after_anomalies_size = 0
        self.after_extradrop_size = 0
        self.final_size = 0
        self.is_droping = False

        if config.configurations['drop']:
            self.is_droping = True
            self.drop_Frame = pd.read_csv(config.inputs['drop_file'])
        else:
            self.drop_Frame = False

    def __find_anomalies(self, data):
        '''
        Find the anomalies (values >= std*3)

        Parameters
        ----------
        data (array of lenght -> n_samples):
        
        Returns
        -------
        anomalies (array of lenght -> n_samples):
            Bool array with outliers marked as 0
        '''
        # Anomalies
        anomalies = np.ones(len(data), dtype=bool)

        # Set upper and lower limit to 3 standard deviation
        std = np.std(data)
        mean = np.mean(data)
        anomaly_cut_off = std * 3
        
        lower_limit  = mean - anomaly_cut_off 
        upper_limit = mean + anomaly_cut_off
        
        # Generate outliers
        for i, value in enumerate(data):
            if value > upper_limit or value < lower_limit:
                anomalies[i] = 0

        return anomalies
    
    def __drop_extra(self, DataFrame, drop_df=False):
        '''
        '''
        if drop_df:
            drop_Frame = drop_df
        else:
            drop_Frame = self.drop_Frame

        ID_to_drop = drop_Frame['ID'].values.tolist()

        for drop in ID_to_drop:
            drop_i = DataFrame.loc[ DataFrame['ID'] == drop ]
            DataFrame = DataFrame.drop(index=drop_i.index)

        return DataFrame

    def __clean_database(self):
        '''
        Clean dataset to perform preparation and splitting
        '''
        dataframe = self.DataFrame

        for target in self.targets:
            # Find anomalies above 3*sigma
            feature_anomalies = self.__find_anomalies(dataframe[target])

            # Remove those anomalies from main_dataframe
            cleaned_dataframe = dataframe[feature_anomalies]
            self.after_anomalies_size = len(cleaned_dataframe)

        return cleaned_dataframe
    
    def load_database(self):
        cleaned_dataframe = self.__clean_database()

        # If drop dataframe is given, will be retired from dataframe
        if self.is_droping:
            cleaned_dataframe = self.__drop_extra(cleaned_dataframe)
            self.after_extradrop_size = len(cleaned_dataframe)

        # Set processed dataframe size
        self.final_size = len(cleaned_dataframe)

        return cleaned_dataframe