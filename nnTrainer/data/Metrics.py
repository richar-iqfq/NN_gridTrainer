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
    
    def __check_finite_values(self, y:np.ndarray, y_pred: np.ndarray):
        return np.isfinite(np.sum(y)) or np.isfinite(np.sum(y_pred))

    def compute(self, y, y_pred):
        y = process_tensor(y)
        y_pred = process_tensor(y_pred)

        # Metrics dictionary
        MAE = {}
        RMSE = {}
        acc = {}
        r2 = {}

        if self.__check_nan_values(y, y_pred) or not self.__check_finite_values(y, y_pred):
            for i, target in enumerate(self.targets):
                MAE[target] = 0
                RMSE[target] = 0
                acc[target] = 0
                r2[target] = 0
                
            MAE['general'] = 0
            RMSE['general'] = 0
            acc['general'] = 0
            r2['general'] = 0

        else:
            try:
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
                
                    if np.isnan(r2_target):
                        r2_target = 0
                    
                    r2[target] = r2_target

                    r2_general += r2_target
                    
                    MAE[target] = mean_absolute_error(y_target, y_pred_target)
                    
                    MSE = mean_squared_error(y_target, y_pred_target)
                    RMSE[target] = np.sqrt(MSE)
                    
                    acc[target] = abs(1 - RMSE[target])
            
                r2['general'] = r2_general/len(self.targets)

            except:
                for i, target in enumerate(self.targets):
                    MAE[target] = 0
                    RMSE[target] = 0
                    acc[target] = 0
                    r2[target] = 0
                
                MAE['general'] = 0
                RMSE['general'] = 0
                acc['general'] = 0
                r2['general'] = 0

        return MAE, RMSE, acc, r2
