import numpy as np
import pandas as pd
import torch

from modules.Configurator import Configurator

class Outliers():
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

    def __assure_is_on_cpu(self, y) -> np.ndarray:
        # Check if y is a tensor
        if torch.is_tensor(y):
            # Move to cpu and detach
            y = y.to('cpu')
            y = y.detach().numpy()

        return y

    def get_numerical_outliers(self, y: np.ndarray, y_pred: np.ndarray, strategy: str='percentual_error', strategy_perc: float=0.08) -> tuple:
        '''
        Get only numerical outliers for ploting purpose
        '''
        y = self.__assure_is_on_cpu(y)
        y_pred = self.__assure_is_on_cpu(y_pred)

        boolean_outliers_dict = {}
        outliers_vals_dict = {}

        for i, target in enumerate(self.targets):
            y_target = y[:,i]
            ytarget_pred = y_pred[:,i]

            # Get stat metrics for different strategies
            if strategy != 'percentual_error':
                # Get difference
                difference = abs(y_target - ytarget_pred)
                err = difference

                # Get stat metrics
                mean = np.mean(difference)
                sigma = np.std(difference)
                diff_max = np.max(difference)

            # ------------ get Outliers -----------------
            if strategy == 'percentual_error':
                tol = strategy_perc*100

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

            elif strategy == 'percentual_difference':
                # Set cut value by a percentage
                cut = diff_max*strategy_perc

                # Get outliers
                boolean_outliers = difference > cut
                
            elif strategy == 'statistical_difference':
                # Set cut value by statistic significance
                cut = mean + 3*sigma

                # Get outliers
                boolean_outliers = difference > cut

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
            for i, target in enumerate(self.targets):
                boolean_target = boolean_outliers[target]

                outliers_ID = ID.loc[boolean_target]

                outliers_dict[f'{target}_ID'] = outliers_ID
                outliers_dict[f'{target}_value'] = outliers_vals[target]

            outliers_df = pd.DataFrame(outliers_dict)
            outliers_df = outliers_df.sort_values('Values', ascending=False)
        except:        
            outliers_df = []

        return boolean_outliers, outliers_df