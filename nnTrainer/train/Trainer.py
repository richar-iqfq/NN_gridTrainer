import os
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import numpy as np
import pandas as pd

from .. import (
    Configurator,
    path_name
)

from nnTrainer.data.Metrics import (
    OutliersComputer,
    MetricsComputer
)

from nnTrainer.data.Database import DatabaseLoader
from nnTrainer.data.Dataset import DatasetBuilder
from nnTrainer.data.Preprocess import PreprocessData
from nnTrainer.data.Plots import PlotsBuilder
from nnTrainer.data.Sql import SqlDatabase
from nnTrainer.tools.Array import (
    process_array,
    is_upper_lower_artifact
)
from nnTrainer.tools.Train import (
    generate_random_string,
    is_model_stuck
)

# Here we import the model trainning
from nnTrainer.train.Models import NetHiddenLayers

#================================ Trainer ===============================================
class Trainer():
    '''
    Main class to create the model, make the data preprocessing and start the training

    Parameters
    ----------
    file_name `str`:
        File name for the results folder.

    '''
    def __init__(self, file_name: str, architecture: dict, hyperparameters: dict, step: str, workers=0):
        # Sql executor
        self.database_executor = SqlDatabase()

        # Path names
        self.path_code = self.get_path_code()
        self.path = os.path.join('Training_results', path_name[step]) # Path to store the results
        self.step = step

        # main config object
        self.config = Configurator()
        
        # Targets
        self.targets = self.config.get_json('targets')

        # Train code
        self.train_code = self.config.get_inputs('train_ID')

        # Backend to run in tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set threads limit to avoid overload of cpu
        if self.config.get_cuda('limit_threads'):
            torch.set_num_threads(1)
        
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # Device type
        self.device = torch.device(self.device_type) # Device
        
        self.file_name = file_name # File where you'll write the results

        # Save plots variable
        self.save_plots = True

        # Architecture variable assignment
        self.num_layers = architecture['num_layers']
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
        self.model = eval(f'NetHiddenLayers({self.num_features}, {self.num_targets}, {self.num_layers}, {self.dimension}, {self.activation_functions})')

        # Build routes
        self.plots_path, self.pred_path = self.__get_routes()

        # To define self.parameters
        self.parameters = self.parameters_count()

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
        
        # Sql executor
        self.database_executor.create_train_record(
            self.num_layers,
            self.dimension,
            self.activation_functions,
            self.parameters,
            self.optim,
            self.crit,
            self.batch_size,
            self.learning_rate,
            self.random_state,
            self.num_epochs,
            step
        )

        # ID and data values (x and y already scaled)
        self.ID, self.x, self.y = self.processer.Retrieve_Processed()

        # datasets
        self.dataset_builder = DatasetBuilder()
        self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_builder.create_datasets()

    def get_path_code(self) -> str:
        '''
        Search if code is not in train results and return the value
        '''
        code = generate_random_string(30)

        for i in range(5):
            if self.database_executor.search_equals('Results', 'Path', code):
                code = generate_random_string(30)
            else:
                break

        return code

    def __get_routes(self):
        '''
        Build database and predictions routes
        '''
        n_layers = str(self.num_layers).zfill(2)

        plots_path = os.path.join(
            self.path,
            self.train_code,
            f'{n_layers}HLayers',
            self.path_code,
        )
        pred_path = os.path.join(plots_path, 'Predictions')
        
        return plots_path, pred_path

    def create_routes(self) -> None:
        '''
        Create plots and predictions folders in storage
        '''
        if not os.path.isdir(self.plots_path):
            os.makedirs(self.plots_path)

        if self.config.get_configurations('save_full_predictions'):
            os.makedirs(self.pred_path)

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
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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

    def write_predictions(self, y, y_pred):
        '''
        write results in csv files
        '''
        values = {
            'ID' : self.ID,
        }

        predictions_file = os.path.join(self.plots_path, 'predictions.csv')

        for i, target in enumerate(self.targets):
            values[target] = y[:,i]
            values[f'{target}_pred'] = y_pred[:,i]

        predictions = pd.DataFrame(values)
        predictions.to_csv(predictions_file, index=False)

    def write_full_predictions(self, x, y_pred):
        '''
        Write full predictions file
        '''
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

    def build_store_values(self, ValidationValues: dict, TestValues: dict, TrainValues: dict, result_values: dict, outliers: dict):
        store_names = ['Val', 'Test', 'Train']
        metrics_names = ['Mae', 'Rmse', 'Acc', 'R2']
        keys = self.targets + ['general']

        store_values = {
            'training_time' : result_values['training_time'],
            'train_loss' : result_values['train_loss'],
            'validation_loss' : result_values['validation_loss'],
            'test_loss' : float(result_values['test_loss'])
        }

        values = {
            'Val' : ValidationValues,
            'Test' : TestValues,
            'Train' : TrainValues,
        }

        # (Val, Test, Train) -> (Mae, Rmse, Acc, R2)
        for name in store_names:
            for metric in metrics_names:
                store_values[f'{metric}{name}_i'] = [values[name][metric][key] for key in keys]

        # Outliers
        store_values['Outliers_i'] = [outliers[f'outliers_{target}'] for target in self.targets]
        store_values['OutliersGeneral'] = outliers['outliers_general']

        return store_values

    def start_training(self, write=True, allow_print=False, save_plots=False, monitoring=False):
        '''
        Runs training
        '''
        # Training flag to check integrity
        train_flag = True
        message = ''

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
        result_values = {}
        loss_train_list = []
        loss_validation_list = []
        general_acc_validation_list = []
        general_acc_train_list = []
        general_r2_validation_list = []
        general_r2_train_list = []

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

                MAE_train, RMSE_train, acc_train, r2_train = self.metrics_calc.compute(y_train, ytrain_pred)

                # Store metrics
                general_acc_train_list.append(acc_train['general'])
                general_r2_train_list.append(r2_train['general'])

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

            ####################################################
            # Check network performance each 50 and 200 epochs #
            ####################################################
            if (epoch+1)%50 == 0:
                if is_model_stuck(general_acc_validation_list, general_r2_validation_list):
                    train_flag = False
                    message = 'Model is stuck!'
                    break

            if (epoch+1)%200 == 0:
                if is_upper_lower_artifact(yval_pred, self.num_targets):
                    train_flag = False
                    message = 'Upper or lower artifact found'
                    break

            # Deep copy model if monitoring
            if monitoring:
                if acc_val['general'] > best_mean_acc:
                    best_general_acc = acc_val['general']
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if allow_print:
                if epoch == 0:
                    print('\n', '#'*37, ' Training Progress ', '#'*37, '\n')

                if (epoch+1)%10 == 0:
                    print(f"Epoch: {(epoch+1):04} Validation: MAE = {MAE_val['general']:.4f} ERR = {RMSE_val['general']:.4f} ACC = {acc_val['general']*100:.2f} r2 = {r2_val['general']:.4f} Loss = {loss_train:.4f}", end='\r')

        if train_flag:
            # =============== Restore best weights when monitoring ===================
            if monitoring:
                if allow_print:
                    print(f'\nBetter performance: epoch = {best_epoch} ___ acc = {best_general_acc}')

                # Update config
                self.config.update(
                    best_acc=best_general_acc,
                    best_epoch=best_epoch
                )

                # Reload weights to model
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

                    MAE_train, RMSE_train, acc_train, r2_train = self.metrics_calc.compute(y_train, ytrain_pred)

                    # Store metrics
                    general_acc_train_list.append(acc_train['general'])
                    general_r2_train_list.append(r2_train['general'])

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
        
            # =========================== Final evaluation ===========================
            with torch.no_grad():
                # ----------------------------- Train ------------------------
                # Unique step
                x_train = copy.deepcopy(x_train)
                y_train = copy.deepcopy(y_train)

                # Load values to device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                with torch.autocast(device_type=self.device_type):
                    ytrain_pred = self.model(x_train)
                    loss_train = criterion(ytrain_pred, y_train)

                # ----------------------------- Test ------------------------
                # Unique step
                x_test = copy.deepcopy(x_testing)
                y_test = copy.deepcopy(y_testing)

                # load values to device
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)

                with torch.autocast(device_type=self.device_type):
                    ytest_pred = self.model(x_test)
                    loss_test = criterion(ytest_pred, y_test)

            # Train
            loss_train = loss_train.item()
            loss_train = torch.as_tensor(loss_train)

            # Test
            loss_test = loss_test.item()
            loss_test = torch.as_tensor(loss_test)

            MAE_train, RMSE_train, acc_train, r2_train = self.metrics_calc.compute(y_train, ytrain_pred)
            MAE_test, RMSE_test, acc_test, r2_test = self.metrics_calc.compute(y_test, ytest_pred)

            et = time.time() # End time
            elapsed_time = et - st

            # Evaluate full data
            x, y, y_pred = self.eval_full_data()

            # Create paths
            self.create_routes()

            # Update metrics dictionary
            result_values['training_time'] = elapsed_time
            result_values['train_loss'] = loss_train.item()
            result_values['validation_loss'] = loss_val.item()
            result_values['test_loss'] = loss_test.numpy()

            # Values to plot
            self.values_plot = {
                'loss_train_list' : loss_train_list,
                'loss_validation_list' : loss_validation_list,
                'general_acc_val' : general_acc_validation_list,
                'general_acc_train' : general_acc_train_list,
                'y_val' : y_val,
                'yval_pred' : yval_pred,
                'y_test' : y_test,
                'ytest_pred' : ytest_pred,
                'y_train' : y_train,
                'ytrain_pred' : ytrain_pred,
                'r2_val' : r2_val,
                'r2_test' : r2_test,
                'r2_train' : r2_train,
                'general_r2_val' : general_r2_validation_list,
                'general_r2_train' : general_r2_train_list
            }

            # Build plots
            self.plots_builder.build_plots(self.values_plot)
            outliers = self.plots_builder.build_full_plots(y, y_pred)

            # Results dict
            validation_values = {
                'Mae' : MAE_val,
                'Rmse' : RMSE_val,
                'Acc' : acc_val,
                'R2' : r2_val
            }

            test_values = {
                'Mae' : MAE_test,
                'Rmse' : RMSE_test,
                'Acc' : acc_test,
                'R2' : r2_test
            }

            train_values = {
                'Mae' : MAE_train,
                'Rmse' : RMSE_train,
                'Acc' : acc_train,
                'R2' : r2_train
            }

            if write:
                model_file = os.path.join(self.plots_path, 'model.pth')
                config_file = os.path.join(self.plots_path, 'config.ini')

                # Save model
                self.save_model(model_file)

                # Metrics
                store_values = self.build_store_values(
                    validation_values,
                    test_values,
                    train_values,
                    result_values,
                    outliers
                )

                self.database_executor.create_result_record(
                    self.plots_path,
                    store_values
                )
            
                # config.ini
                self.write_config(config_file)

                # Predictions
                self.write_predictions(y, y_pred)

                # Full predictions
                if self.config.get_configurations('save_full_predictions'):
                    self.write_full_predictions(x, y_pred)

        return train_flag, message