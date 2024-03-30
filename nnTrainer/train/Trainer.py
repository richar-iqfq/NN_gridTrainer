import os
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np

from .. import Configurator
from nnTrainer.data.Metrics import (
    OutliersComputer,
    MetricsComputer,
    Writter
)
from nnTrainer.data.Database import DatabaseLoader
from nnTrainer.data.Dataset import DatasetBuilder
from nnTrainer.data.Preprocess import PreprocessData
from nnTrainer.data.Plots import PlotsBuilder
from nnTrainer.tools.Array import process_array

# Here we import the different models for trainning
from nnTrainer.train.Models import (
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
    def __init__(self, file_name, architecture, hyperparameters, workers=0, step=None):
        # Path names
        self.path_name = {
            'explore_lr' : '00_explore_lr',
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
        self.config = Configurator()
        
        # Targets
        self.targets = self.config.get_json('targets')

        # Backend to run in tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set threads limit to avoid overload of cpu
        if self.config.get_cuda('limit_threads'):
            torch.set_num_threads(1)
        
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # Device type
        self.device = torch.device(self.device_type) # Device
        self.path = os.path.join('Training_results') # Path to store the results
        
        if step:
            self.path = os.path.join(self.path, self.path_name[step])
            self.step = step

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        
        self.file_name = file_name # File where you'll write the results

        # Result values
        self.result_values = {}

        # Save plots variable
        self.save_plots = True

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

        self.random_state = self.config.get_custom('random_state')
        self.workers = workers
        self.is_plots = False

        # Model definition
        self.model = eval(f'{self.model_name}({self.num_features}, {self.num_targets}, {self.dimension}, {self.activation_functions})')

        # Build routes
        self.plots_path, self.pred_path = self.__build_routes()

        # To define self.parameters
        self.parameters_count() 

        # Outliers values per target
        self.outliers_count = {}
        
        # Drop molecules
        if self.config.get_configurations('drop'):
            self.drop = self.config.get_inputs('drop_file')
        else:
            self.drop = False

        self.standard_line = [
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

        # Loader Classes
        self.processer = PreprocessData()
        self.loader = DatabaseLoader()
        self.outliers_calc = OutliersComputer()
        self.metrics_calc = MetricsComputer()
        self.plots_builder = PlotsBuilder(self.plots_path)
        self.writter = Writter(self.path, self.file_name, self.plots_path)

        # ID and data values (x and y already scaled)
        self.ID, self.x, self.y = self.processer.Retrieve_Processed()

        # datasets
        self.dataset_builder = DatasetBuilder()
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_builder.create_datasets()

    def __build_routes(self):
        dim = str(self.dimension).replace(', ', '|')
        arch = str(self.activation_functions).replace(', ', '|')
        general_folder = f'{arch}_{dim}'

        step_name = {
            'grid' : '',
            'optimization' : f'{self.optim}_{self.crit}',
            'tuning_batch' : f'batches_{self.batch_size}',
            'tuning_lr' : f'lr_{self.learning_rate}',
            'explore_lr' : f'lr_{self.learning_rate}',
            'recovering' : f'lr_{self.learning_rate}',
            'random_state' : f"rs_{self.config.get_custom('random_state')}",
            'around_exploration' : f"ae_{self.learning_rate}_{self.batch_size}"
        }

        file_name = self.file_name

        plots_path = os.path.join(self.path, 'Plots', file_name.replace('.csv', ''), general_folder, step_name[self.step])

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
        Returns
        -------
        size `int`:
            Number of samples used in training, size = n_samples*n_targets
        '''
        size = len(self.train_dataset) * self.num_targets
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

    def show_plots(self):
        '''
        Show the final plots of the training
        '''
        self.plots_builder.show_plots()

    def close_plots(self):
        '''
        Close the active plots
        '''
        self.plots_builder.close_plots()
    
    def eval_full_data(self):
        x, y = copy.deepcopy(self.x), copy.deepcopy(self.y)

        x = process_array(x)
        y = process_array(y)

        self.model = self.model.to('cpu')

        with torch.no_grad():
            y_pred = self.model(x)

        return x, y, y_pred

    def write_config(self, path):
        '''
        Write config.ini file on path
        '''
        self.config.save_ini(path)

    def __save_full_predictions(self, x, y_pred):
        _, y_unscaled = self.processer.Unscale(x, y_pred)

        df = self.loader.load_database()

        for i, target in enumerate(self.targets):
            df[f'{target}_pred'] = y_unscaled[:,i].flatten()

        df.to_csv(os.path.join(self.pred_path, self.file_name.replace('.csv', '_FPredictions.csv')), index=False)

    def __instance_Dataloaders(self):
        # Dataloader
        if self.batch_size == 'All':
            bz = len(self.train_dataset)
        else:
            bz = int(self.batch_size)

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
        
    def is_model_stuck(self, acc, r2, yval_pred):

        if np.mean(acc[-15::]) == 0:
            return True

        if np.mean(r2[-5::]) == 1:
            return True

        for i, target in enumerate(self.targets):
            if yval_pred[:,i].mean() == 0:
                return True

        return False

    def start_training(self, write=True, allow_print=False, save_plots=False, monitoring=False):
        '''
        Runs training
        '''
        # Set saving plots variable
        self.save_plots = save_plots

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
        general_acc_validation_list = []
        general_r2_validation_list = []

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

                with torch.autocast(device_type=self.device_type):
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

                with torch.autocast(device_type=self.device_type):
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

                with torch.autocast(device_type=self.device_type):
                    yval_pred = self.model(x_val)
                    loss_val = criterion(yval_pred, y_val)

                loss_validation_list.append(loss_val.item())

                MAE_val, RMSE_val, acc_val, r2_val = self.metrics_calc.compute(y_val, yval_pred)

                # Store metrics
                general_acc_validation_list.append(acc_val['general'])
                general_r2_validation_list.append(r2_val['general'])

            # Check if model is stuck each 50 epochs
            if (epoch+1)%50 == 0:
                if self.is_model_stuck(general_acc_validation_list, general_r2_validation_list, yval_pred):
                    raise Exception('Model is stuck!')

            # Deep copy the model if monitoring
            if monitoring:
                if acc_val['general'] > best_mean_acc:
                    best_general_acc = acc_val['general']
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if allow_print:
                if epoch == 0:
                    print('\n', '#'*37, ' Training Progress ', '#'*37, '\n')

                if (epoch+1)%10 == 0:
                    print(f"Epoch: {(epoch+1):04} Validation: MAE = {MAE_val['general']:.4f} ERR = {RMSE_val['general']:.4f} ACC = {acc_val['general']*100:.2f} r2 = {r2_val['general']:.4f}", end='\r')

        # ===== Restore best weights when monitoring =====
        if monitoring:
            # print(f'\nBetter performance: epoch = {best_epoch} ___ acc = {best_general_acc}')

            self.config.update(
                best_acc=best_general_acc,
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

                with torch.autocast(device_type=self.device_type):
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

                with torch.autocast(device_type=self.device_type):
                    yval_pred = self.model(x_val)
                    loss_val = criterion(yval_pred, y_val)

                loss_validation_list.append(loss_val.item())

                MAE_val, RMSE_val, acc_val, r2_val = self.metrics_calc.compute(y_val, yval_pred)

                # Store metrics
                general_acc_validation_list.append(acc_val['general'])
                general_r2_validation_list.append(r2_val['general'])
    
        # ----------------------------- Test ------------------------
        with torch.no_grad():
            # Unique step
            x_test = copy.deepcopy(x_testing)
            y_test = copy.deepcopy(y_testing)

            # load values to device
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)

            with torch.autocast(device_type=self.device_type):
                ytest_pred = self.model(x_test)
                loss_test = criterion(ytest_pred, y_test)

        loss_test = loss_test.item()
        loss_test = torch.as_tensor(loss_test)

        MAE_test, RMSE_test, acc_test, r2_test = self.metrics_calc.compute(y_test, ytest_pred)

        et = time.time() # End time
        elapsed_time = et - st

        # Update metrics dictionary
        self.result_values['training_time'] = elapsed_time
        self.result_values['train_loss'] = loss_train.item()
        self.result_values['validation_loss'] = loss_val.item()
        self.result_values['test_loss'] = loss_test.numpy()
        
        for target in self.targets + ['general']:
            self.result_values[f'MAE_val_{target}'] = MAE_val[target]
            self.result_values[f'MAE_test_{target}'] = MAE_test[target]
            self.result_values[f'RMSE_val_{target}'] = RMSE_val[target]
            self.result_values[f'RMSE_test_{target}'] = RMSE_test[target]
            self.result_values[f'acc_val_{target}'] = acc_val[target]
            self.result_values[f'acc_test_{target}'] = acc_test[target]
            self.result_values[f'r2_val_{target}'] = r2_val[target]
            self.result_values[f'r2_test_{target}'] = r2_test[target]

        self.values_plot = {
            'loss_train_list' : loss_train_list,
            'loss_validation_list' : loss_validation_list,
            'general_acc_val' : general_acc_validation_list,
            'y_val' : y_val,
            'yval_pred' : yval_pred,
            'y_test' : y_test,
            'ytest_pred' : ytest_pred,
            'r2_val' : r2_val,
            'r2_test' : r2_test,
            'general_r2_val' : general_r2_validation_list
        }

        # Evaluate full data
        x, y, y_pred = self.eval_full_data()

        # Build plots
        self.plots_builder.build_plots(self.values_plot)
        self.outliers = self.plots_builder.build_full_plots(y, y_pred)

        if write:
            model_file = os.path.join(self.plots_path, 'model.pth')
            config_file = os.path.join(self.plots_path, 'config.ini')

            # Save model
            self.save_model(model_file)

            # Metrics
            self.writter.write_metrics(self.standard_line, self.result_values, self.outliers)
            
            # Predictions
            self.writter.write_predictions(self.ID, y, y_pred)
        
            # config.ini
            self.write_config(config_file)

            # Full predictions
            if self.config.get_configurations('save_full_predictions'):
                self.__save_full_predictions(x, y_pred)