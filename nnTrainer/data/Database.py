import numpy as np
import pandas as pd

from .. import Configurator

#==================================================================
#======================= Database Loader ==========================
#==================================================================
class DatabaseLoader():
    def __init__(self):
        '''
        Main class for the loading and cleaning of data.
        '''
        self.config: Configurator = Configurator()

        # Variable assingment
        self.features = self.config.get_json('features')
        self.targets = self.config.get_json('targets')

        self.DataFrame = pd.read_csv(self.config.database_path)

        self.initial_size = len(self.DataFrame)
        self.after_unconverged_size = 0
        self.after_anomalies_size = 0
        self.after_extradrop_size = 0
        self.final_size = 0
        self.is_droping = False

        if self.config.get_configurations('drop'):
            self.is_droping = True
            self.drop_Frame = pd.read_csv(self.config.get_inputs('drop_file'))
        else:
            self.drop_Frame = False

    def __remove_b_not_converged(self, DataFrame: pd.DataFrame) -> pd.DataFrame:
        '''
        Remove the non-converged molecules

        Parameters
        ----------
        DataFrame (obj of class pd.DataFrame):
            Fresh DataFrame to be filtered

        Returns
        ------
        filtered (obj of class pd.DataFrame):
            DataFrame without the non-converged values
        '''
        convergence_crit = 1e-6

        target = self.targets[0]
        b = int(target[-1])

        filtered = DataFrame[ DataFrame[f'Err{b}'] <= convergence_crit ]

        return filtered
    
    def __remove_b_extreme_values(self, DataFrame: pd.DataFrame, b: int=False) -> pd.DataFrame:
        '''
        Remeve the extreme b values, like those above 100 or bellow -100

        Parameters
        ----------
        DataFrame (obj of class pd.DataFrame):
            DataFrame with extreme values

        b (optional) (int):
            b scope to remove values, default self.b

        Returns
        -------
        filtered (obj of class pd.DataFrame):
            DataFrame without the extreme values

        '''
        target = self.targets[0]

        max_limit = 100
        min_limit = -100

        filtered = DataFrame[ DataFrame[target] >= min_limit ]
        filtered = filtered[ filtered[target] <= max_limit ]

        return filtered

    def __remove_b_statistical_anomalies(self, DataFrame: pd.DataFrame) -> pd.DataFrame:
        '''
        Remove statistical anomalies (values above or bellow std*3)

        Parameters
        ----------
        DataFrame (obj of class pd.DataFrame):
            DataFrame with data

        Returns
        -------
        cleaned_dataframe (obj of class pd.DataFrame):
            DataFrame without the statiscal anomalies (those above or bellow 3 times sigma)
        '''
        target = self.targets[0]
        # Set data
        data = DataFrame[target]

        # Set upper and lower limit to /3/ 5 standard deviation
        std = np.std(data)
        mean = np.mean(data)
        anomaly_cut_off = std * 5
        
        lower_limit  = mean - anomaly_cut_off 
        upper_limit = mean + anomaly_cut_off
        
        # Filter values
        cleaned_dataframe = DataFrame[ DataFrame[target] >= lower_limit ]
        cleaned_dataframe = cleaned_dataframe[ cleaned_dataframe[target] <= upper_limit ]

        return cleaned_dataframe

    def __find_anomalies(self, data):
        '''
        Find the anomalies (values >= std*3)

        Parameters
        ----------
        data (array of lenght -> n_samples):
        
        Returns
        -------
        anomalies (array of lenght -> n_samples):
            Boolean array with outliers marked as 0
        '''
        # Anomalies
        anomalies = np.ones(len(data), dtype=bool)

        # Set upper and lower limit to 3 times the standard deviation
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
        Drop the specified IDs (drop file) from DataFrame
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
        trainID = self.config.get_inputs('train_ID')

        if 'b' == trainID[0]:
            # Remove not converged molecules
            preprocessed_dataframe = self.__remove_b_not_converged(dataframe)

            # Remove the extreme values like 100 or -100
            preprocessed_dataframe = self.__remove_b_extreme_values(preprocessed_dataframe)

            # Remove anomalies above and bellow 3*sigma
            cleaned_dataframe = self.__remove_b_statistical_anomalies(preprocessed_dataframe)

        else:
            for target in self.targets:
                # Find anomalies above 3*sigma
                feature_anomalies = self.__find_anomalies(dataframe[target])

                # Remove those anomalies from main_dataframe
                cleaned_dataframe = dataframe[feature_anomalies]
                self.after_anomalies_size = len(cleaned_dataframe)

        return cleaned_dataframe
    
    def load_database(self):
        '''
        Loads the DataFrame to forward computation.
        '''
        # Clean dataset by removing anomalies
        cleaned_dataframe = self.__clean_database()

        # If drop dataframe is given, those IDs will be retired from dataframe
        if self.is_droping:
            cleaned_dataframe = self.__drop_extra(cleaned_dataframe)
            self.after_extradrop_size = len(cleaned_dataframe)

        # Set processed dataframe size
        self.final_size = len(cleaned_dataframe)

        return cleaned_dataframe
