import os

import pandas as pd

from .. import Configurator

class Recorder():
    '''
    Class to save and read training scripts info
    '''
    def __init__(self) -> None:
        self.record_path = 'logs/runs.csv'
        self.config: Configurator = Configurator()

        # Check if file exists
        if os.path.isfile(self.record_path):
            self.record = self.__load_record()
        else:
            print('Record file not found! Writting new file...')
            self.__create_record()
    
    def __load_record(self) -> pd.DataFrame:
        return pd.read_csv(self.record_path)
    
    def __create_record(self) -> None:
        if not os.path.isdir('logs'):
            os.makedirs('logs')

        data = {
            'code': ['A000'],
            'scaled': [False],
            'lineal_output': [True],
            'steps_order' : ['']
        }
        
        dataframe = pd.DataFrame(data)
        dataframe.to_csv(self.record_path, index=False)

        self.record = self.__load_record()
    
    def save_values(self, perform: list) -> None:
        data = {
            'code': [ self.config.get_inputs('train_ID') ],
            'scaled': [ self.config.get_inputs('scale_y') ],
            'lineal_output': [ self.config.get_inputs('lineal_output') ],
            'steps_order' : [ str(perform) ]
        }

        if data['code'] not in self.record['code'].values:
            record = pd.concat([self.record, pd.DataFrame(data)], ignore_index=True)
            record.to_csv(self.record_path, index=False)

            self.record = self.__load_record()
    
    def get_values(self, train_ID: str) -> tuple:
        '''
        Read training values from record file

        Parameters
        ----------
        train_ID `str`:
            Training ID

        Returns
        -------
        b `int`:
            b value
        alpha `float`:
            alpha value
        scale_y `bool`:
            True if output values were scaled
        '''
        data = self.record[self.record['code'] == train_ID]

        if len(data) == 0:
            raise Exception('Train ID not found')

        b = data['b'].values[0]
        alpha = data['alpha'].values[0]
        scale_y = data['scaled'].values[0]

        return b, alpha, scale_y