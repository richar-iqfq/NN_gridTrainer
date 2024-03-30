import os

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from .. import Configurator
from nnTrainer.tools.Array import process_tensor

class OutliersComputer():
    '''
    Class for Outliers computing.

    Parameters
    ----------
    percent (optional) : `float`
        Outliers percent cut off
    '''
    def __init__(self, percent: float=None) -> None:
        self.config: Configurator = Configurator()

        self.targets = self.config.get_json('targets')
        
        if percent:
            self.percent = percent
        else:
            self.percent = self.config.get_configurations('percent_outliers')

        self.alpha = 3 # Anomalies cut

    def get_difference(self, y_target: np.ndarray, ytarget_pred: np.ndarray):
        '''
        Compute difference y_target - ytarget_pred
        '''
        return abs(y_target - ytarget_pred)

    def percentual_error(self, y_target: np.ndarray, ytarget_pred: np.ndarray, ):
        '''
        Get outliers by percentual error strategy
        '''
        tol = self.percent*100

        # Get percentual error
        if y_target.all():
            err = abs( (y_target - ytarget_pred)/(y_target) )*100

            # Get boolean array
            boolean_outliers = err > tol

        else:
            err = np.zeros_like(y_target)

            for i, y_value in enumerate(y_target):
                if y_value != 0:
                    err[i] = abs( (y_value - ytarget_pred[i])/(y_value) )*100
                else:
                    err[i] = np.inf

                boolean_outliers = err > tol

        return err, boolean_outliers
    
    def percentual_difference(self, y_target: np.ndarray, ytarget_pred: np.ndarray):
        '''
        Get outliers by percentual difference strategy
        '''
        # Get difference
        difference = self.get_difference(y_target, ytarget_pred)

        diff_max = np.max(difference)

        # Set cut value by a percentage
        cut = diff_max*self.percent

        # Get outliers
        boolean_outliers = difference > cut

        return difference, boolean_outliers

    def statistical_difference(self, y_target: np.ndarray, ytarget_pred: np.ndarray):
        '''
        Get outliers by statistical difference strategy
        '''
        # Get difference
        difference = self.get_difference(y_target, ytarget_pred)
        
        # Get stat metrics
        mean = np.mean(difference)
        sigma = np.std(difference)

        # Set cut value by statistic significance
        cut = mean + self.alpha*sigma

        # Get outliers
        boolean_outliers = difference > cut

        return difference, boolean_outliers

    def get_numerical_outliers(self, y: np.ndarray, y_pred: np.ndarray, strategy: str='percentual_error') -> tuple:
        '''
        Get only numerical outliers for ploting purpose
        '''
        y = process_tensor(y)
        y_pred = process_tensor(y_pred)

        boolean_outliers_dict = {}
        outliers_vals_dict = {}

        for i, target in enumerate(self.targets):
            y_target = y[:,i]
            ytarget_pred = y_pred[:,i]

            # ------------ get Outliers -----------------
            if strategy == 'percentual_error':
                err, boolean_outliers = self.percentual_error(y_target, ytarget_pred)

            elif strategy == 'percentual_difference':
                err, boolean_outliers = self.percentual_difference(y_target, ytarget_pred)
                
            elif strategy == 'statistical_difference':
                err, boolean_outliers = self.statistical_difference(y_target, ytarget_pred)

            else:
                raise Exception('Invalid strategy requested...')

            outliers_vals = err[boolean_outliers]

            outliers_vals_dict[target] = outliers_vals
            boolean_outliers_dict[target] = boolean_outliers

        return boolean_outliers_dict, outliers_vals_dict

    def get_total_outliers(self, ID: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> tuple:
        '''
        Get outlier values and dict
        '''
        strategy = self.config.get_configurations('outliers_strategy')

        boolean_outliers, outliers_vals = self.get_numerical_outliers(y, y_pred, strategy=strategy)

        outliers_dict = {}

        try:
            for target in self.targets:
                boolean_target = boolean_outliers[target]

                outliers_ID = ID.loc[boolean_target]

                outliers_dict[f'{target}_ID'] = outliers_ID
                outliers_dict[f'{target}_value'] = outliers_vals[target]

            outliers_df = pd.DataFrame(outliers_dict)
            outliers_df = outliers_df.sort_values('Values', ascending=False)
        except:        
            outliers_df = []

        return boolean_outliers, outliers_df
    
class MetricsComputer():
    def __init__(self) -> None:
        self.config = Configurator()
        self.alpha = 5 # Shift to avoid zero values
        self.targets = self.config.get_json('targets')

    def __check_nan_values(self, y: np.ndarray, y_pred: np.ndarray):
        return np.isnan(np.sum(y)) or np.isnan(np.sum(y_pred))

    def compute(self, y, y_pred):
        y = process_tensor(y)
        y_pred = process_tensor(y_pred)

        # Metrics dictionary
        MAE = {}
        RMSE = {}
        acc = {}
        r2 = {}

        if self.__check_nan_values(y, y_pred):
            for i, target in enumerate(self.targets):
                MAE[target] = 1
                RMSE[target] = 1
                acc[target] = 0
                r2[target] = 1
                
            MAE['general'] = 1
            RMSE['general'] = 1
            acc['general'] = 0
            r2['general'] = 1

        else:
            MAE['general'] = mean_absolute_error(y, y_pred)
            
            MSE = mean_squared_error(y, y_pred)
            RMSE['general'] = np.sqrt(MSE)
            
            acc['general'] = abs(1 - RMSE['general'])

            # Initialize r2 general
            r2_general = 0
            
            # For target metrics
            for i, target in enumerate(self.targets):
                y_target = y[:,i]
                y_pred_target = y_pred[:,i]

                r2_target = np.corrcoef(y_target+self.alpha, y_pred_target+self.alpha)[0,1]**2
                r2[target] = r2_target if r2_target else 0

                r2_general += r2_target
                
                MAE[target] = mean_absolute_error(y_target, y_pred_target)
                
                MSE = mean_squared_error(y_target, y_pred_target)
                RMSE[target] = np.sqrt(MSE)
                
                acc[target] = abs(1 - RMSE[target])
        
            r2['general'] = r2_general/len(self.targets)

        return MAE, RMSE, acc, r2
    
class Writter():
    def __init__(self, path: str, file_name: str, plots_path: str) -> None:
        self.config = Configurator()

        self.targets = self.config.get_json('targets')
        self.metrics_file = os.path.join(path, file_name)
        self.predictions_file = os.path.join(plots_path, 'predictions.csv')

        self.standard_names, self.result_names, self.total_names = self.__get_column_names()

    def __get_column_names(self):
        standard_column_names = [ 
            'dimension', 'architecture', 'parameters', 
            'optimizer', 'loss_function', 'epochs',
            'batch_size', 'lr', 'random_state'
        ]

        training_column_names = [
            'training_time', 'train_loss', 'validation_loss', 'test_loss'
        ]

        MAE_val_column_names = [f'MAE_val_{target}' for target in self.targets]
        MAE_test_column_names = [f'MAE_test_{target}' for target in self.targets]

        RMSE_val_column_names = [f'RMSE_val_{target}' for target in self.targets]
        RMSE_test_column_names = [f'RMSE_test_{target}' for target in self.targets]

        acc_val_column_names = [f'acc_val_{target}' for target in self.targets]
        acc_test_column_names = [f'acc_test_{target}' for target in self.targets]
        
        r2_val_column_names = [f'r2_val_{target}' for target in self.targets] 
        r2_test_column_names = [f'r2_test_{target}' for target in self.targets]
        
        general_val_column_names = ['MAE_val_general', 'RMSE_val_general', 'acc_val_general', 'r2_val_general']
        general_test_column_names = ['MAE_test_general', 'RMSE_test_general', 'acc_test_general', 'r2_test_general']
        outliers_column_names = [f'outliers_{target}' for target in self.targets] + ['outliers_general']

        # Sum all the lists
        results_column_names = training_column_names + MAE_val_column_names + MAE_test_column_names + \
                                RMSE_val_column_names + RMSE_test_column_names + acc_val_column_names + \
                                acc_test_column_names + r2_val_column_names + r2_test_column_names + \
                                general_val_column_names + general_test_column_names + outliers_column_names

        # Total column
        total_column_names = standard_column_names + results_column_names

        return standard_column_names, results_column_names, total_column_names

    def __check_metrics_file(self):
        if not os.path.isfile(self.metrics_file):
            with open(self.metrics_file, 'w') as txt:
                txt.write(','.join(self.total_names))

    def write_metrics(self, standard_line: list, result_values: dict, outliers: dict):
        '''
        Save all the results inside csv file
        '''
        values = {**result_values, **outliers}

        self.__check_metrics_file()

        results_line = []
        for column in self.result_names:
            if 'acc' in column:
                value = np.round(values[column]*100, 2)
            else:
                value = np.round(values[column], 4)

            results_line.append(value)

        final_string = [str(f) for f in standard_line + results_line]

        line = ','.join(final_string)
        with open(self.metrics_file, 'a') as txt:
            txt.write(f'\n{line}')

    def write_predictions(self, ID, y, y_pred):
        '''
        write results in csv files
        '''
        values = {
            'ID' : ID,
        }

        for i, target in enumerate(self.targets):
            values[target] = y[:,i]
            values[f'{target}_pred'] = y_pred[:,i]

        predictions = pd.DataFrame(values)
        predictions.to_csv(self.predictions_file, index=False)