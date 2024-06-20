import os

import numpy as np
import pandas as pd

from .. import Configurator
from nnTrainer.base_class.Singleton import SingletonMeta
from nnTrainer.base_class.SQlite import SQlite3Executor

class SqlDatabase(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.config = Configurator()
        self.tables = [
            'Launchs',
            'Trains',
            'Results'
        ]

        self.columns = {}

        self.LastIDs = {
            'LaunchID' : '',
            'TrainID' : '',
            'ResultID' : ''
        }

        database_name = self.config.get_inputs('database')
        self.database_name = database_name.replace('csv', 'db')

        self.database_path = os.path.join('Training_results', self.database_name)

        if not os.path.isfile(self.database_path):
            self.prepare_database()
        else:
            self.SQlite_executor = SQlite3Executor(self.database_path)

        for table in self.tables:
            self.columns[table] = self.get_columns(table)

    def prepare_database(self) -> None:
        self.SQlite_executor = SQlite3Executor(self.database_path)

        script_file = 'nnTrainer/tools/databaseScript.txt'

        if os.path.isfile(script_file):
            with open(script_file, 'r') as script:
                lines = script.readlines()
                lines = ''.join(lines)
                query = lines.split(';')
        else:
            raise Exception('No database creation script found!')

        for instruction in query:
            self.SQlite_executor.execute_simple(instruction)
        
        self.SQlite_executor.commit()

    def retrieve_last_id(self, table: str) -> None:
        # Retrieve last inserted id
        self.SQlite_executor.execute_simple(f"SELECT last_insert_rowid() FROM {table}")

        id = self.SQlite_executor.retrieve()
        return id[0][0]

    def get_columns(self, table):
        columns = []
        query = f"PRAGMA table_info({table});"
        
        self.SQlite_executor.execute_simple(query)
        values = self.SQlite_executor.retrieve()

        for value in values:
            columns.append(value[1])

        return columns

    def create_launch_record(self, steps: list) -> None:
        train_ID = self.config.get_inputs('train_ID')
        create_record = True

        exists = self.search_equals('Launchs', 'Code', train_ID)
        # Check if record exists
        if exists:
            print(f'{train_ID} code already exists')
            rewrite = input('Delete all data and create new record? [y/n]: ')

            if rewrite == 'y':
                # Delete record
                self.delete_equals('Launchs', 'Code', train_ID)

                # Enable writing
                create_record = True
            else:
                # Retrive record ID
                Launc_ID = exists[0][0]
                self.LastIDs['LaunchID'] = Launc_ID
                # Disable writting
                create_record = False
        
        if create_record:
            # Write record
            values = (
                None,
                train_ID,
                self.config.get_configurations('min_neurons'),
                self.config.get_configurations('max_neurons'),
                self.config.get_configurations('n_tries'),
                self.config.get_configurations('n_networks'),
                self.config.get_configurations('reader_criteria'),
                self.config.get_configurations('percent_outliers'),
                self.config.get_configurations('network_tolerance'),
                self.config.get_configurations('drop'),
                self.config.get_configurations('config_file'),
                self.config.get_configurations('outliers_strategy'),
                self.config.get_custom('lineal_output'),
                self.config.get_custom('seed'),
                str(self.config.get_json('features')),
                str(self.config.get_json('targets')),
                str(steps),
            )

            query_line = "INSERT INTO Launchs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

            self.SQlite_executor.execute_parameters(query_line, values)

            self.SQlite_executor.commit()

            self.LastIDs['LaunchID'] = self.retrieve_last_id('Launchs')

    def create_train_record(self, num_hidden: int, dimension: str, activation: str, parameters: int, optimizer: str, lossFunction: str, batch_size: int, learning_rate: float, random_state: int, epochs: int, step: str) -> None:
        if batch_size == 'All':
            batch_size = 0
        
        values = (
            None,
            int(num_hidden),
            str(dimension),
            str(activation),
            parameters,
            str(optimizer),
            str(lossFunction),
            int(batch_size),
            float(learning_rate),
            int(random_state),
            int(epochs),
            str(step),
            self.LastIDs['LaunchID']
        )
        
        query_line = "INSERT INTO Trains VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

        self.SQlite_executor.execute_parameters(query_line, values)

        self.SQlite_executor.commit()

        self.LastIDs['TrainID'] = self.retrieve_last_id('Trains')
    
    def create_result_record(self, path: str, results: dict) -> None:
        values = (
            None,
            results['training_time'],
            results['train_loss'],
            results['validation_loss'],
            results['test_loss'],
            str(results['MaeVal_i']),
            str(results['RmseVal_i']),
            str(results['AccVal_i']),
            str(results['R2Val_i']),
            str(results['MaeTest_i']),
            str(results['RmseTest_i']),
            str(results['AccTest_i']),
            str(results['R2Test_i']),
            str(results['MaeTrain_i']),
            str(results['RmseTrain_i']),
            str(results['AccTrain_i']),
            str(results['R2Train_i']),
            str(results['Outliers_i']),
            results['OutliersGeneral'],
            path,
            self.LastIDs['TrainID'],
        )

        query_line = "INSERT INTO Results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

        self.SQlite_executor.execute_parameters(query_line, values)

        self.SQlite_executor.commit()

        self.LastIDs['ResultID'] = self.retrieve_last_id('Results')

    def search_equals(self, table: str, column: str, value: str) -> list:
        query = f"SELECT * FROM {table} WHERE {column} == '{value}'"
        
        self.SQlite_executor.execute_simple(query)

        values = self.SQlite_executor.retrieve()

        return values
    
    def search_in(self, table: str, column: str, values: tuple) -> list:
        query = f"SELECT * FROM {table} WHERE {column} IN {tuple(values)}"

        self.SQlite_executor.execute_simple(query)

        values = self.SQlite_executor.retrieve()

        return values

    def delete_equals(self, table: str, column: str, value: str) -> None:
        query = f"DELETE FROM {table} WHERE {column} == '{value}'"

        self.SQlite_executor.execute_simple(query)
        self.SQlite_executor.commit()

    def purge_results(self, Launch: str):
        pass

    def close_database(self):
        self.SQlite_executor.close()

class SqlReader():
    '''
    Better neural network aschitecture reader
    '''
    def __init__(self):
        self.config = Configurator()
        self.SQLdatabase = SqlDatabase()
    
    def map_data_to_dataframe(self, table: str, values: str) -> dict:
        if len(values)==0:
            raise Exception(f'Results for {table} are empty!')
        
        mapped_data = {}
        values = np.array(values)
        values = np.transpose(values)

        for i, column in enumerate(self.SQLdatabase.columns[table]):
            mapped_data[column] = values[i]

        return pd.DataFrame(mapped_data)

    def get_launch_data(self, LaunchCode: str) -> dict:
        launch_values =  self.SQLdatabase.search_equals('Launchs', 'Code', LaunchCode)

        data = self.map_data_to_dataframe('Launchs', launch_values)

        return data
    
    def get_train_data(self, LaunchID: int, NumLayers: int, Step: str) -> dict:
        train_values = self.SQLdatabase.search_equals('Trains', 'LaunchID', LaunchID)

        data = self.map_data_to_dataframe('Trains', train_values)

        data = data[data.Step==Step]
        data = data[data.NumLayers==str(NumLayers)]

        if data.empty:
            raise Exception(f'Train values for {NumLayers} Layers in {Step} empty')

        return data
    
    def get_results_data(self, TrainIDs: list) -> dict:
        results_values = self.SQLdatabase.search_in('Results', 'TrainID', TrainIDs)

        data = self.map_data_to_dataframe('Results', results_values)

        return data
    
    def retrieve_values(self, LaunchCode: str, NumLayers: int, Step: str) -> pd.DataFrame:
        launch_data = self.get_launch_data(LaunchCode)
        launch_id = launch_data.LaunchID[0]

        train_data = self.get_train_data(launch_id, NumLayers, Step)
        train_ids = train_data['TrainID'].values

        result_data = self.get_results_data(train_ids)

        return train_data, result_data
    
    def sort_values(self, result_data: pd.DataFrame, criteria: str, worst: bool) -> pd.DataFrame:
        # Sort by criteria
        if 'Outliers' in criteria:
            asc = True
            result_data['criteria'] = result_data[criteria]

        else:
            crit = []
            asc = False
            
            for i, cell in enumerate(result_data[criteria]):
                values = eval(cell)
                # Last element in Metric refers to general evaluation
                crit.append(values[-1])

            result_data['criteria'] = crit

        if worst:
            asc = not asc
        
        return result_data.sort_values(['criteria'], ignore_index=True, ascending=asc)
    
    def filter_values(self, result_data: pd.DataFrame, n_values: int) -> pd.DataFrame:
        index_to_drop = []
        
        for metric in ['R2Val_i', 'R2Test_i', 'AccVal_i', 'AccTest_i']:
            for i, cell in enumerate(result_data[metric]):
                values = eval(cell)
                # Last element in Metric refers to general evaluation
                value = values[-1]

                if value == 0:
                    index_to_drop.append(i)

        if len(index_to_drop) > 0:
            result_data = result_data.drop(index_to_drop)

        return result_data.head(n_values)

    def build_best(self, train_data: pd.DataFrame, filtered_result_data: pd.DataFrame) -> list:
        better_networks = []
        
        for index, result_row in filtered_result_data.iterrows():
            acc_test_list = eval(result_row['AccTest_i'])
            r2_test_list = eval(result_row['R2Test_i'])

            acc_val_list = eval(result_row['AccVal_i'])
            r2_val_list = eval(result_row['R2Val_i'])

            train_id = result_row['TrainID']
            train_row = train_data[train_data.TrainID == train_id]

            better_networks.append(
                {
                    'hidden_layers' : int(train_row['NumLayers'].values[0]),
                    'dimension' : eval(train_row['Dimension'].values[0]),
                    'activation_functions' : eval(train_row['Activation'].values[0]),
                    'optimizer' : train_row['Optimizer'].values[0],
                    'criterion' : train_row['LossFunction'].values[0],
                    'random_state' : int(train_row['RandomState'].values[0]),
                    'num_epochs' : int(train_row['Epochs'].values[0]),
                    'batch_size' : int(train_row['BatchSize'].values[0]),
                    'lr' : float(train_row['Lr'].values[0]),
                    'parameters' : int(train_row['Parameters'].values[0]),
                    'acc_test' : float(acc_test_list[-1]),
                    'r2_test' : float(r2_test_list[-1]),
                    'acc_val' : float(acc_val_list[-1]),
                    'r2_val' : float(r2_val_list[-1]),
                    'outliers' : int(result_row['OutliersGeneral']),
                    'Path' : result_row['Path']
                }
            )

        return better_networks

    def recover_best(self, LaunchCode: str, NumLayers: int, Step: str, criteria='R2Val_i', n_values=1, worst=False) -> dict:
        if criteria not in ('R2Val_i', 'R2Test_i', 'OutliersGeneral', 'AccTest_i', 'AccVal_i'):
            raise Exception('Wrong criteria value')
        
        # Retrieve initial data
        train_data, result_data = self.retrieve_values(LaunchCode, NumLayers, Step)

        # sort values by criteria
        sorted_result_data = self.sort_values(result_data, criteria, worst)

        # Filter 0 values from acc and r2
        filtered_result_data = self.filter_values(sorted_result_data, n_values)

        # Create best network dict
        best_network = self.build_best(train_data, filtered_result_data)

        return best_network