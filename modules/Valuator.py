import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import copy

from modules.PreprocessData import PreprocessData
from modules.DatasetBuilder import create_datasets
from modules.ResultsReader import Reader
from modules.DatabaseLoader import DatabaseLoader

# Here we import the different models for trainning
from modules.Models import (
    Net_1Hlayer,
    Net_2Hlayer,
    Net_3Hlayer,
    Net_4Hlayer,
    Net_5Hlayer,
    Net_6Hlayer
)

#====================================== Valuator ==============================================
class Valuator():
    def __init__(self, config, hidden_layers=4, step='random_state', reader_criteria='outliers_count', mode='complete', workers=0):
        # Path names
        self.path_name = {
            'grid' : '01_grid',
            'optimization' : '02_optimization',
            'tuning_batch' : '03_tuning_batch',
            'tuning_lr' : '04_tuning_lr',
            'lineal' : '05_lineal',
            'random_state' : '06_random_state',
            'around_exploration' : '07_around_exploration',
            'recovering' : 'recovering' 
        }
        
        # main config object
        self.config = config
        # extra route
        self.step = step
        # Targets
        self.targets = self.config.json['targets']
        self.num_targets = self.config.json['num_targets']
        # Features
        self.features = self.config.json['features']
        self.num_features = self.config.json['num_features']

        # Backend to run in tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if config.cuda['limit_threads']:
            torch.set_num_threads(1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device

        self.results_path = os.path.join('Training_results', mode, self.path_name[step])

        self.Networks = {
            1 : 'Net_1Hlayer',
            2 : 'Net_2Hlayer',
            3 : 'Net_3Hlayer',
            4 : 'Net_4Hlayer',
            5 : 'Net_5Hlayer',
            6 : 'Net_6Hlayer'
        }

        self.file_name = config.custom['extra_filename']
        self.names = [ 
            'dimension', 'architecture', 'parameters', 'optimizer', 'loss_function', 'epochs',
            'batch_size', 'lr', 'training_time', 'random_state', 'train_loss', 'val_loss',
            'test_loss', 'mae_val', 'mae_test', 'err_val', 'err_test', 'r2_val', 'r2_test',
            'acc_val', 'acc_test', 'outliers_count'
        ] # Column names for the results file

        # Preprocesser
        self.processer = PreprocessData(self.config)

        # Loader
        self.loader = DatabaseLoader(self.config)

        # Get better network
        self.netReader = Reader(hidden_layers, self.file_name, type='complete', step=self.step)

        # Architecture variable assignment
        self.model_name = self.Networks[hidden_layers]
        self.get_architecture(reader_criteria)

        # Get the model saved file path
        self.model_file = self.find_directory()

        print(self.model_file)

        # Model definition
        self.model = eval(f"{self.model_name}({self.num_features}, {self.num_targets}, {self.dimension}, {self.activation_functions})")

        # Outliers value
        self.outliers_count = 0
        self.Outliers_DF = []

        if config.configurations['drop']:
            self.drop = config.inputs['drop_file']
        else:
            self.drop = False

        # datasets
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(self.processer)

        # Load model
        self.load_model()
    
    def find_directory(self):
        dim = str(self.dimension).replace(', ', '|')
        arch = str(self.activation_functions).replace(', ', '|')
        general_folder = f'{arch}_{dim}'

        results_csv = self.netReader.files[0].split('/')[-1].replace('.csv', '')

        step_name = {
            'grid' : '',
            'optimization' : f'{self.optimizer}_{self.criterion}',
            'tuning_batch' : f'batches_{self.batch_size}',
            'tuning_lr' : f'lr_{self.learning_rate}',
            'recovering' : f'lr_{self.learning_rate}',
            'random_state' : f"rs_{self.config.custom['random_state']}",
            'around_exploration' : f"ae_{self.learning_rate}_{self.batch_size}"
        }

        file_name = results_csv

        if self.step != 'tuning_lr':
            plots_path = os.path.join(self.results_path, 'Plots', file_name, general_folder, step_name[self.step])
            model_file = os.path.join(plots_path, 'model.pth')
        else:
            lr_path = os.path.join(self.results_path, 'Plots', file_name, general_folder)
            for folder in os.listdir(lr_path):
                if str(self.learning_rate) in folder:
                    plots_path = os.path.join(lr_path, folder)
                    model_file = os.path.join(plots_path, 'model.pth')

                    break

        return model_file

    def get_architecture(self, reader_criteria):
        nets = self.netReader.recover_best(1, reader_criteria)
        network = nets[0]

        self.config.update(
            random_state = network['random_state']
        )

        for key in network:
            print(f'{key}  ->  {network[key]}')

        self.batch_size = network['batch_size']
        self.learning_rate = network['lr']

        self.dimension = network['dimension']
        self.activation_functions = network['activation_functions']
        self.optimizer = network['optimizer']
        self.criterion = network['criterion']
        self.random_state = network['random_state']

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()

    def eval_model(self, x, y):
        criterion = eval(self.criterion)
        
        x = torch.from_numpy(x)
        x = x.to(torch.float32)

        y = torch.from_numpy(y)
        y = y.to(torch.float32)

        with torch.no_grad():
            x_val = copy.deepcopy(x)

            y_pred = self.model(x_val)
            loss_val = criterion(y_pred, y)

        return y_pred, loss_val

    def __get_outliers(self, y, y_pred):
        percent = self.config.configurations['percent_outliers']
        tol = percent*100

        boolean_outliers = {}

        for i, target in enumerate(self.targets):
            y_target = y[:,i]
            ytarget_pred = y_pred[:,i]

            boolean = np.zeros(len(y_target), dtype=bool)
            diff = np.zeros(len(y_target))
            
            for j, y_value in enumerate(y_target):
                if y_value != 0:
                    err = abs( (y_value - ytarget_pred[j])/(y_value) )*100
                else:
                    err = np.inf

                if err > tol:
                    boolean[j] = True
                    diff[j] = err

            boolean_outliers[target] = boolean
        
        return boolean_outliers
    
    def __compute_metrics(self, y, y_pred):
        y = y.detach().numpy()
        y_pred = y_pred.detach().numpy()

        # Get the correlation factor
        lineal = LinearRegression()

        # Metrics dictionary
        MAE = {}
        RMSE = {}
        acc = {}
        r2 = {}

        # General metrics
        lineal.fit(y, y_pred)
        r2['general'] = lineal.score(y, y_pred)

        MAE['general'] = mean_absolute_error(y, y_pred)
        
        MSE = mean_squared_error(y, y_pred)
        RMSE['general'] = np.sqrt(MSE)
        
        acc['general'] = abs(1 - RMSE['general'])

        # For target metrics
        for i, target in enumerate(self.targets):
            y_target = y[:,i].reshape(-1, 1)
            y_pred_target = y_pred[:,i].reshape(-1, 1)

            lineal.fit(y_target, y_pred_target)
            
            r2[target] = lineal.score(y_target, y_pred_target)
            
            MAE[target] = mean_absolute_error(y_target, y_pred_target)
            
            MSE = mean_squared_error(y_target, y_pred_target)
            RMSE[target] = np.sqrt(MSE)
            
            acc[target] = abs(1 - RMSE[target])
        
        return MAE, RMSE, acc, r2

    def build_plot(self, y, y_pred, boolean_outliers):
        y = torch.from_numpy(y)
        y = y.to(torch.float32)

        MAE, RMSE, acc, r2 = self.__compute_metrics(y, y_pred)

        # Regression plots
        for i, target in enumerate(self.targets):
            # Variable assingment
            y_target = y[:,i].flatten()
            ytarget_pred = y_pred[:,i].flatten()

            # Plot
            fig_full, ax = plt.subplots(1)
            fig_full.suptitle(f'Full Regression {target}', weight='bold')
            fig_full.set_size_inches(20, 13)
            
            outliers_count = np.count_nonzero(boolean_outliers[target])
            
            ax.set_title(f'r2 = {r2[target]:.4f}      outliers count = {outliers_count}', weight='bold')
            
            ax.set_xlabel('y')
            ax.set_ylabel('y_pred')

            textstr = '\n'.join([
                f'MAE = {MAE[target]:.4f}',
                f'RMSE = {RMSE[target]:.4f}',
                f'acc = {acc[target]*100:.2f}'
                ])
            
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)

            ax.plot(y_target, y_target, '-b')
            ax.scatter(y_target[boolean_outliers[target]], ytarget_pred[boolean_outliers[target]], color='yellowgreen')
            ax.plot(y_target, ytarget_pred, '.r')

    def show_plot(self):
        plt.show()

    def run(self, show=True):
        ID, x, y = self.processer.Retrieve_Processed()

        x = copy.deepcopy(x)
        y = copy.deepcopy(y)
        
        y_pred, _ = self.eval_model(x, y)

        boolean_outliers = self.__get_outliers(y, y_pred.numpy())

        if self.config.configurations['save_full_predictions']:
            pd.set_option('mode.chained_assignment', None)

            dataFrame = self.loader.load_database()
            
            path = os.path.join(self.results_path, 'Predictions')
            file = os.path.join(path, f'{self.model_name}{self.file_name}_FP.csv')

            _, y_unscaled = self.processer.Unscale(x, y_pred)

            for i, target in enumerate(self.targets):
                y_unscaled_pred = y_unscaled[:,i].flatten()

                dataFrame[f'{target}_pred'] = y_unscaled_pred
            
            dataFrame.to_csv(file, index=False)

        self.build_plot(y, y_pred, boolean_outliers)

        if show:
            self.show_plot()