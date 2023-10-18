import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import pandas as pd
import numpy as np
import copy
import re

from modules.PreprocessData import PreprocessData
from modules.DatabaseLoader import DatabaseLoader
from modules.DatasetBuilder import create_datasets

# Here we import the different models for trainning
from modules.Models import (
    Net_1Hlayer,
    Net_2Hlayer,
    Net_3Hlayer,
    Net_4Hlayer,
    Net_5Hlayer,
    Net_6Hlayer
)

#================================ Trainer ===============================================
class Trainer():
    '''
    Main class to create the model, make the data preprocessing and start the training

    Parameters
    ----------
    file_name `str`:
        File name for the results folder.

    '''
    def __init__(self, file_name, architecture, hyperparameters, config, mode='complete', workers=0, extra_route=None):
        # main config object
        self.config = config
        
        # Backend to run in tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set threads limit to avoid overload of cpu
        if config.cuda['limit_threads']:
            torch.set_num_threads(1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device

        self.path = os.path.join('Training_results', mode) # Path to store the results
        
        if extra_route:
            self.path = os.path.join(self.path, extra_route)
            self.extra_route = extra_route

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        
        self.mode = mode # Level of training (complete, recovering)
        self.file_name = file_name # File where you'll write the results
        
        # Dictionary to store the training results
        self.standard_column_names, self.results_column_names = self.__get_column_names()

        # Result values
        self.result_values = {}

        # Architecture variable assignment
        self.model_name = architecture['model']
        self.num_features = architecture['num_features']
        self.num_targets = architecture['num_targets']
        self.dimension = architecture['dimension']
        self.activation_functions = architecture['activation_functions']
        self.optim = architecture['optimizer']
        self.crit = architecture['criterion']

        # Hyperparameters variable assignment
        self.num_epochs = hyperparameters['num_epochs']
        self.batch_size = hyperparameters['batch_size']
        self.learning_rate = hyperparameters['learning_rate']

        self.workers = workers
        self.is_plots = False

        # Model definition
        self.model = eval(f'{self.model_name}({self.num_features}, {self.num_targets}, {self.dimension}, {self.activation_functions})')

        # Build routes
        self.plots_path, self.pred_path = self.__build_routes()

        # Outliers values per target
        self.outliers_count = {}
        self.Outliers_DF = {}

        # Drop molecules
        if config.configurations['drop']:
            self.drop = config.inputs['drop_file']
        else:
            self.drop = False

        # Loader Classes
        self.processer = PreprocessData(config)
        self.loader = DatabaseLoader(config)

        # ID and data values (x and y already scaled)
        self.ID, self.x, self.y = self.processer.Retrieve_Processed()

        # datasets
        self.random_state = config.custom['random_state']
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(self.processer)

    def __get_column_names(self):
        standard_column_names = [ 
            'dimension', 'architecture', 'parameters', 
            'optimizer', 'loss_function', 'epochs',
            'batch_size', 'lr', 'random_state']

        training_column_names = ['training_time', 'train_loss', 'validation_loss', 'test_loss']

        MAE_val_column_names = [f'MAE_val_{target}' for target in self.config.json['targets']] + ['MAE_val_mean']
        MAE_test_column_names = [f'MAE_test_{target}' for target in self.config.json['targets']] + ['MAE_test_mean']

        MSE_val_column_names = [f'MSE_val_{target}' for target in self.config.json['targets']] + ['MSE_val_mean']
        MSE_test_column_names = [f'MSE_test_{target}' for target in self.config.json['targets']] + ['MSE_test_mean']

        acc_val_column_names = [f'acc_val_{target}' for target in self.config.json['targets']] + ['acc_val_mean']
        acc_test_column_names = [f'acc_test_{target}' for target in self.config.json['targets']] + ['acc_test_mean']
        
        r2_val_column_names = [f'r2_val_{target}' for target in self.config.json['targets']] + ['r2_val_mean']
        r2_test_column_names = [f'r2_test_{target}' for target in self.config.json['targets']] + ['r2_test_mean']
        
        # outliers_column_names = [f'outliers_{target}' for target in self.config.json['targets']]

        # Sum all the lists
        results_column_names = training_column_names + MAE_val_column_names + MAE_test_column_names + \
                                MSE_val_column_names + MSE_test_column_names + acc_val_column_names + \
                                acc_test_column_names + r2_val_column_names + r2_test_column_names #+ \
                                # outliers_column_names

        return standard_column_names, results_column_names

    def __build_routes(self):
        dim = str(self.dimension).replace(', ', '|')
        arch = str(self.activation_functions).replace(', ', '|')
        general_folder = f'{arch}_{dim}'

        extra_route_name = {
            'grid' : '',
            'optimization' : f'{self.optim}_{self.crit}',
            'tuning_batch' : f'batches_{self.batch_size}',
            'tuning_lr' : f'lr_{self.learning_rate}',
            'recovering' : f'lr_{self.learning_rate}',
            'random_state' : f"rs_{self.config.custom['random_state']}",
            'around_exploration' : f"ae_{self.learning_rate}_{self.batch_size}"
        }

        # Check if filename contains mode($n)
        not_allowed_keys = {
            'tuning_batch' : r'batches\d',
            'tuning_lr' : r'lr\d',
            'around_exploration' : r'RE\d'
        }

        replacement = {
            'tuning_batch' : r'batches',
            'tuning_lr' : r'lr',
            'around_exploration' : r'RE'
        }

        if self.extra_route in ['tuning_batch', 'tuning_lr', 'around_exploration']:
            if 'lineal' in self.file_name:
                file_name = self.file_name

            if re.search(not_allowed_keys[self.extra_route], self.file_name):
                file_name = re.sub(not_allowed_keys[self.extra_route], replacement[self.extra_route], self.file_name)
        else:
            file_name = self.file_name

        plots_path = os.path.join(self.path, 'Plots', file_name.replace('.csv', ''), general_folder, extra_route_name[self.extra_route])

        pred_path = os.path.join(self.path, 'Predictions')

        if not os.path.isdir(pred_path):
            os.makedirs(pred_path)

        if not os.path.isdir(plots_path):
            os.makedirs(plots_path)

        return plots_path, pred_path

    def overview(self):
        '''
        Print a table with all the network's information (parameters, architecture, etc)
        '''
        print('')
        print('#'*37, 'Model Overview', '#'*37)
        summary(self.model, (1,  self.num_features))

        print(f'Train Molecules {len(self.train_dataset)}')
        print(f'Test Molecules: {len(self.test_dataset)}')
        print(f'Val Molecules: {len(self.val_dataset)}\n')

    def database_size(self):
        '''
        Return the number of samples used in training
        '''
        size = len(self.train_dataset)
        return size

    def parameters_count(self):
        '''
        Return the amount of trainable parameters in network
        '''
        param = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.parameters = param

        return param

    def reset(self):
        '''
        Reset parameters
        '''
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def save_model(self, file):
        '''
        Save the model (state_dict) to file
        '''
        torch.save(self.model.state_dict(), file)

    def load_model(self, file):
        '''
        Load the model (state_dict) from file
        '''
        self.model.load_state_dict(torch.load(file))
        print(self.model.state_dict())

        print(f'Model Loaded from: {file}\n')

    def state(self):
        '''
        Print the state_dict of the model
        '''
        print('Model State:')
        print(self.model.state_dict(), '\n')

    def __build_plots(self, save=False):        
        '''
        Build all the required plots
        '''
        # Loss graphic
        fig1, ax1 = plt.subplots(1)
        fig1.set_size_inches(20, 13)

        ax1.plot(self.values_plot['loss_train_list'], '-r')
        ax1.plot(self.values_plot['loss_validation_list'], ':g')
        ax1.legend(['Train', 'Validation'])
        ax1.set_title('Loss', weight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')

        # Regression plot
        y_val = self.values_plot['y_val']
        yval_pred = self.values_plot['yval_pred']

        y_test = self.values_plot['y_test']
        ytest_pred = self.values_plot['ytest_pred']

        # Detach and move to cpu
        y_val = y_val.to('cpu')
        yval_pred = yval_pred.to('cpu')

        y_val = y_val.detach().numpy()
        yval_pred = yval_pred.detach().numpy()

        y_test = y_test.to('cpu')
        ytest_pred = ytest_pred.to('cpu')

        y_test = y_test.detach().numpy()
        ytest_pred = ytest_pred.detach().numpy()

        r2_val = self.values_plot['r2_val']
        r2_test = self.values_plot['r2_test']

        fig2, ax = plt.subplots(2)
        fig2.suptitle('Regression', weight='bold')
        fig2.set_size_inches(20, 13)

        y = [y_val, y_test]
        r2 = [r2_val[-1], r2_test]
        y_pred = [yval_pred, ytest_pred]
        label = ['Validation', 'Test']
        style = ['.r', '.m']
        for i in range(2):
            boolean_outliers, _ = self.__get_outliers(y[i], y_pred[i])
            
            ax[i].plot(y[i], y[i], '-b')
            ax[i].scatter(y[i][boolean_outliers], y_pred[i][boolean_outliers], color='yellowgreen')
            ax[i].plot(y[i], y_pred[i], style[i])
            
            ax[i].set_xlabel('y')
            ax[i].set_ylabel('y_pred')

            ax[i].set_title(f'{label[i]} r2 = {r2[i]:.4f}      outliers count = {np.count_nonzero(boolean_outliers) }', weight='bold')

        # accuracy plots
        fig3, ax3 = plt.subplots(1)
        fig3.set_size_inches(20, 13)

        ax3.plot(self.values_plot['acc_val'], '--g')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy %')

        ax3.set_title('Accuracy', weight='bold')
        
        # Regression over epochs
        fig4, ax4 = plt.subplots(1)
        fig4.set_size_inches(20, 13)

        ax4.plot(self.values_plot['r2_val'])
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('r2')

        ax4.set_title('Regression coeficient', weight='bold')

        self.is_plots = True

        if save:
            fig1.savefig(os.path.join(self.plots_path, 'loss.pdf'), dpi=450, format='pdf')
            fig2.savefig(os.path.join(self.plots_path, 'corr.pdf'), dpi=450, format='pdf')
            fig3.savefig(os.path.join(self.plots_path, 'acc.pdf'), dpi=450, format='pdf')
            fig4.savefig(os.path.join(self.plots_path, 'reg.pdf'), dpi=450, format='pdf')

            # self.__save_img_outliers()

    def __build_full_plots(self, y, y_pred, save=True):
        # ========================================== Scaled Plot =====================================================0
        # Regression plot
        fig_full, ax = plt.subplots(1)
        fig_full.suptitle('Full Regression', weight='bold')
        fig_full.set_size_inches(20, 13)

        boolean_outliers, _ = self.__get_outliers(y, y_pred)
        
        MAE, RMSE, acc, r2, linear_predicted = self.__compute_metrics(y, y_pred)
        
        ax.plot(y, y, '-b')
        ax.plot(y, linear_predicted, '-g')
        ax.scatter(y[boolean_outliers], y_pred[boolean_outliers], color='yellowgreen')
        ax.plot(y, y_pred, '.r')
        
        ax.set_xlabel('y')
        ax.set_ylabel('y_pred')

        ax.set_title(f'r2 = {r2:.4f}      outliers count = {np.count_nonzero(boolean_outliers) }', weight='bold')

        textstr = '\n'.join([
            f'MAE = {MAE:.4f}',
            f'RMSE = {RMSE:.4f}',
            f'acc = {acc*100:.2f}'
            ])
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        if save:
            fig_full.savefig(os.path.join(self.plots_path, 'full_corr.pdf'), dpi=450, format='pdf')

        del(fig_full)

        # ================================== Unscaled Plot ===========================================
        if self.config.inputs['scale_y'] == True or self.config.custom['lineal_output'] == False:
            # Regression plot
            fig_full, ax = plt.subplots(1)
            fig_full.suptitle('Full Regression over b', weight='bold')
            fig_full.set_size_inches(20, 13)

            b = self.processer.y_unscale_routine(y)
            b_pred = self.processer.y_unscale_routine(y_pred)

            boolean_outliers, _ = self.__get_outliers(b, b_pred)
            
            MAE, RMSE, acc, r2, linear_predicted = self.__compute_metrics(b, b_pred)
            
            ax.plot(b, b, '-b')
            ax.plot(b, linear_predicted, '-g')
            ax.scatter(b[boolean_outliers], b_pred[boolean_outliers], color='yellowgreen')
            ax.plot(b, b_pred, '.r')
            
            ax.set_xlabel('b')
            ax.set_ylabel('b_pred')

            ax.set_title(f'r2 = {r2:.4f}      outliers count = {np.count_nonzero(boolean_outliers) }', weight='bold')

            textstr = '\n'.join([
                f'MAE = {MAE:.4f}',
                f'RMSE = {RMSE:.4f}',
                f'acc = {acc*100:.2f}'
                ])
            
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            if save:
                fig_full.savefig(os.path.join(self.plots_path, 'full_bcorr.pdf'), dpi=450, format='pdf')

    def show_plots(self, save=False):
        '''
        Show the final plots of the training
        '''
        if not self.is_plots:
            self.__build_plots(save=save)
        
        plt.show()

    def close_plots(self):
        '''
        Close the active plots
        '''
        plt.close('all')
    
    def write_metrics(self):
        '''
        Save all the results inside csv file
        '''
        columns = self.standard_column_names + self.results_column_names
        values = self.result_values

        if not os.path.isfile(os.path.join(self.path, self.file_name)):
            with open(os.path.join(self.path, self.file_name), 'w') as txt:
                txt.write(','.join(columns))
        
        standard_line = [
            str(self.dimension).replace(',', '|'),
            str(self.activation_functions).replace(', ', '|'),
            self.parameters,
            self.optim,
            self.crit,
            self.num_epochs,
            self.batch_size,
            self.learning_rate,
            self.random_state
        ]

        results_line = []
        for column in self.results_column_names:
            if 'acc' in column:
                value = np.round(values[column]*100, 2)
            else:
                value = np.round(values[column], 4)

            results_line.append(value)

        final_string = [str(f) for f in standard_line + results_line]

        line = ','.join(final_string)
        with open(os.path.join(self.path, self.file_name), 'a') as txt:
            txt.write(f'\n{line}')
    
    def write_predictions(self):
        '''
        write results in csv files
        '''
        ID, x, y = copy.deepcopy(self.ID), copy.deepcopy(self.x), copy.deepcopy(self.y)

        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        x = x.to('cpu')

        y = torch.from_numpy(y)
        y = y.to(torch.float32)
        y = y.to('cpu')

        self.model = self.model.to('cpu')

        with torch.no_grad():
            y_pred = self.model(x)

        values = {
            'ID' : ID,
            'y' : y.flatten(),
            'y_pred' : y_pred.flatten()
        }

        self.__build_full_plots(y, y_pred)
        _, outliers_df = self.__get_outliers(y, y_pred)

        self.outliers_count = len(outliers_df)
        self.Outliers_DF = outliers_df

        predictions = pd.DataFrame(values)
        predictions.to_csv(os.path.join(self.plots_path, 'predictions.csv'), index=False)
        outliers_df.to_csv(os.path.join(self.plots_path, 'outliers.csv'), index=False)

        if self.config.configurations['save_full_predictions']:
            self.__save_full_predictions(x, y_pred)

    def write_config(self, path):
        '''
        Write config.ini file on path
        '''
        self.config.save_ini(path)

    def __save_full_predictions(self, x, y_pred):
        _, y_unscaled = self.processer.Unscale(x, y_pred)

        df = self.loader.load_database()
        df['b_pred'] = y_unscaled.flatten()

        df.to_csv(os.path.join(self.pred_path, self.file_name.replace('.csv', '_FPredictions.csv')), index=False)

    def __instance_Dataloaders(self):
        # Dataloader
        if self.batch_size == 'All':
            bz = len(self.train_dataset)
        else:
            bz = self.batch_size

        train_loader = DataLoader(dataset=self.train_dataset,
                            batch_size=bz,
                            num_workers=self.workers,
                            shuffle=True)
        
        train_loader_full = DataLoader(dataset=self.train_dataset, 
                                       batch_size=len(self.train_dataset),
                                       num_workers=self.workers,
                                       shuffle=False)
        
        val_loader = DataLoader(dataset=self.val_dataset,
                                batch_size=len(self.val_dataset),
                                num_workers=self.workers,
                                shuffle=False)

        test_loader = DataLoader(dataset=self.test_dataset,
                            batch_size=len(self.test_dataset),
                            num_workers=self.workers,
                            shuffle=False)
        
        return train_loader, train_loader_full, val_loader, test_loader

    def __compute_metrics(self, y, y_pred):
        try:
            y = y.to('cpu')
            y_pred = y_pred.to('cpu')

            y = y.detach().numpy()
            y_pred = y_pred.detach().numpy()
        except:
            pass

        # Get the correlation factor
        lineal = LinearRegression()

        # Metrics dictionary
        MAE = {}
        MSE = {}
        acc = {}
        r2 = {}
        linear_predicted = {}

        if np.isnan(np.sum(y)) or np.isnan(np.sum(y_pred)):
            for i, target in enumerate(self.config.json['targets']):
                MAE[target] = 1
                MSE[target] = 1
                acc[target] = 0
                r2[target] = 1
                
                MAE['mean'] = 1
                MSE['mean'] = 1
                acc['mean'] = 0
                r2['mean'] = 1

                linear_predicted = np.ones_like(y[:,i])

        else:
            for i, target in enumerate(self.config.json['targets']):
                y_target = y[:,i].reshape(-1, 1)
                y_pred_target = y_pred[:,i].reshape(-1, 1)

                lineal.fit(y_target, y_pred_target)
                
                r2[target] = lineal.score(y_target, y_pred_target)
                linear_predicted[target] = lineal.predict(y_target)

                MAE[target] = mean_absolute_error(y_target, y_pred_target)
                MSE[target] = mean_squared_error(y_target, y_pred_target)
                
                acc[target] = abs(1 - MSE[target])
            
        # Compute means
        MAE['mean'] = np.mean( list(MAE.values()) )
        MSE['mean'] = np.mean( list(MSE.values()) )
        acc['mean'] = np.mean( list(acc.values()) )
        r2['mean'] = np.mean( list(r2.values()) )
        
        return MAE, MSE, acc, r2, linear_predicted

    def __get_outliers(self, y, y_pred):
        boolean_outliers = np.zeros(len(y), dtype=bool)
        diff_outliers = np.zeros(len(y))
        ID = self.ID
        percent = self.config.configurations['percent_outliers']

        for i, y_value in enumerate(y):
            tol = percent*100

            if y_value != 0:
                err = abs( (y_value - y_pred[i])/(y_value) )*100
            else:
                err = np.inf

            if err > tol:
                boolean_outliers[i] = True
                diff_outliers[i] = err

        outliers_vals = diff_outliers[boolean_outliers]

        try:
            outliers_ID = ID.loc[boolean_outliers]
            
            outliers_df = pd.DataFrame(
                {'ID' : outliers_ID,
                'Values' : outliers_vals
                }
            )

            outliers_df = outliers_df.sort_values('Values', ascending=False)
        except:
            outliers_df = []

        return boolean_outliers, outliers_df
        
    def start_training(self, write=True, allow_print=False, save_plots=False, monitoring=False):
        '''
        Runs training
        '''
        # load model to device
        self.model.to(self.device)

        # Instance dataloaders
        train_loader, train_loader_full, val_loader, test_loader = self.__instance_Dataloaders()
        
        # Training full tensor with all the data (batch_size == len(train set))
        for _, (x_tr, y_tr) in enumerate(train_loader_full):
            x_train_full, y_train_full = x_tr, y_tr

        # Validation tensors
        for _, (x_v, y_v) in enumerate(val_loader):
            x_validation, y_validation = x_v, y_v

        # Test tensors
        for _, (x_t, y_t) in enumerate(test_loader):
            x_testing, y_testing = x_t, y_t

        # Optimizer definition
        optimizer = eval(f'torch.optim.{self.optim}(self.model.parameters(), lr={self.learning_rate})')

        # Criterion definition
        criterion = eval(self.crit)

        # Define the metric lists
        loss_train_list = []
        loss_validation_list = []
        mean_acc_validation_list = []
        mean_r2_validation_list = []

        # ======================================================================
        # ========================== Training loop =============================
        # ======================================================================
        st = time.time() # Start time

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_mean_acc = 0.0

        for epoch in range(self.num_epochs):

            for _, (x_train, y_train) in enumerate(train_loader): # Here we use train_loader due to batch size
                # Load values to device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.autocast(device_type='cuda'):
                    # Forward pass
                    ytrain_pred = self.model(x_train)
                    loss_train = criterion(ytrain_pred, y_train)

                # Backward and optimize
                loss_train.backward()
                optimizer.step()

            # |_________________________________________________________________|
            # |__________________________ Metrics ______________________________|
            # |_________________________________________________________________|
            with torch.no_grad():
                # -----------------------Training------------------------------
                # Unique step
                x_train = copy.deepcopy(x_train_full)
                y_train = copy.deepcopy(y_train_full)
                
                # Load values to device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                with torch.autocast(device_type='cuda'):
                    ytrain_pred = self.model(x_train)
                    loss_train = criterion(ytrain_pred, y_train)

                loss_train_list.append(loss_train.item())

                # ---------------------Validation------------------------------
                # Unique step
                x_val = copy.deepcopy(x_validation)
                y_val = copy.deepcopy(y_validation)

                # load values to device
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)

                with torch.autocast(device_type='cuda'):
                    yval_pred = self.model(x_val)
                    loss_val = criterion(yval_pred, y_val)

                loss_validation_list.append(loss_val.item())

                MAE_val, MSE_val, acc_val, r2_val, _ = self.__compute_metrics(y_val, yval_pred)

                # Store metrics
                mean_acc_validation_list.append(acc_val['mean'])
                mean_r2_validation_list.append(r2_val['mean'])

            # Check if model is stuck each 50 epochs
            if (epoch+1)%50 == 0:
                if np.mean(mean_acc_validation_list[-15::]) == 0:
                    break

            # Deep copy the model if monitoring
            if monitoring:
                if acc_val['mean'] > best_mean_acc:
                    best_mean_acc = acc_val['mean']
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if allow_print:
                if epoch == 0:
                    print('\n', '#'*37, ' Training Progress ', '#'*37, '\n')

                if (epoch+1)%10 == 0:
                    print(f"Epoch: {(epoch+1):04} Validation: mean_MAE = {MAE_val['mean']:.4f} ERR = {MSE_val['mean']:.4f} mean_ACC = {acc_val['mean']*100:.2f} mean_r2 = {r2_val['mean']:.4f}", end='\r')

        # ===== Restore best when monitoring =====
        if monitoring:
            print(f'\nBetter performance: epoch = {best_epoch} ___ mean_acc = {best_mean_acc}')

            self.config.update(
                best_mean_acc=best_mean_acc,
                best_epoch=best_epoch
            )
            self.model.load_state_dict(best_model_wts)

            # Recompute steps
            with torch.no_grad():
                # -----------------------Training------------------------------
                # Unique step
                x_train = copy.deepcopy(x_train_full)
                y_train = copy.deepcopy(y_train_full)
                
                # Load values to device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                with torch.autocast(device_type='cuda'):
                    ytrain_pred = self.model(x_train)
                    loss_train = criterion(ytrain_pred, y_train)

                loss_train_list.append(loss_train.item())

                # ---------------------Validation------------------------------
                # Unique step
                x_val = copy.deepcopy(x_validation)
                y_val = copy.deepcopy(y_validation)

                # load values to device
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)

                with torch.autocast(device_type='cuda'):
                    yval_pred = self.model(x_val)
                    loss_val = criterion(yval_pred, y_val)

                loss_validation_list.append(loss_val.item())

                MAE_val, MSE_val, acc_val, r2_val, _ = self.__compute_metrics(y_val, yval_pred)

                # Store metrics
                mean_acc_validation_list.append(acc_val['mean'])
                mean_r2_validation_list.append(r2_val['mean'])
    
        # ----------------------------- Test ------------------------
        with torch.no_grad():
            # Unique step
            x_test = copy.deepcopy(x_testing)
            y_test = copy.deepcopy(y_testing)

            # load values to device
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)

            with torch.autocast(device_type='cuda'):
                ytest_pred = self.model(x_test)
                loss_test = criterion(ytest_pred, y_test)

        loss_test = loss_test.item()
        loss_test = torch.as_tensor(loss_test)

        MAE_test, MSE_test, acc_test, r2_test, _ = self.__compute_metrics(y_test, ytest_pred)

        et = time.time() # End time
        elapsed_time = et - st

        self.parameters_count() # To define self.parameters

        # Update metrics dictionary
        self.result_values['training_time'] = elapsed_time
        self.result_values['train_loss'] = loss_train.item()
        self.result_values['validation_loss'] = loss_val.item()
        self.result_values['test_loss'] = loss_test.numpy()
        
        for target in self.config.json['targets'] + ['mean']:
            self.result_values[f'MAE_val_{target}'] = MAE_val[target]
            self.result_values[f'MAE_test_{target}'] = MAE_test[target]
            self.result_values[f'MSE_val_{target}'] = MSE_val[target]
            self.result_values[f'MSE_test_{target}'] = MSE_test[target]
            self.result_values[f'acc_val_{target}'] = acc_val[target]
            self.result_values[f'acc_test_{target}'] = acc_test[target]
            self.result_values[f'r2_val_{target}'] = r2_val[target]
            self.result_values[f'r2_test_{target}'] = r2_test[target]

        self.values_plot = {
            'loss_train' : loss_train_list,
            'loss_validation' : loss_validation_list,
            'mean_acc_val' : mean_acc_validation_list,
            'y_val' : y_val,
            'yval_pred' : yval_pred,
            'y_test' : y_test,
            'ytest_pred' : ytest_pred,
            'r2_val' : r2_val,
            'r2_test' : r2_test,
            'mean_r2_val' : mean_r2_validation_list
        }

        if write:
            self.save_model(os.path.join(self.plots_path, 'model.pth'))
            self.write_predictions()
            self.write_metrics()
            self.write_config(os.path.join(self.plots_path, 'config.ini'))

        if save_plots:
            self.__build_plots(save=True)