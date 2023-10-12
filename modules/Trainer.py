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
    Main training class to create the model, make the data preprocessing and start the training

    Attributes
    ----------
    file_name : `str`
        File name where the results are going to be written inside the path: 
        Training_results/`mode`/

    architecture : `dict`
        Dictionary with the whole network architecture: model (`class name of models`) (str),
        num_features (int), dimension (`list`) (int), activation_functions (`list`) (str),
        optimizer (`class name of nn.OPTIMIZERS`) (str), criterion (`class name of nn.CRITERION`) (str)

    hyperparameters : `dict`
        Dictionary with the hyperparameters to train the network: num_epochs (`int`), batch_size
        (`int` or All), learning_rate (`int`)

    config : instance of class `Configurator`
        Object with all the main configurations and parameters to be used during training

    mode : `str`
        Mode of training (soft, tuning, strong, specific), default is soft

    workers : 'int'
        num_workers for the dataloader in the trainning, default is 4

    extra_route : 'str'
        if want to store inside an specific folder in path Training_results/mode/extra_route

    Methods
    -------
    start_training(write=True, save_plots=False, allow_print=False)
        Main method to start the training

    overview()
        Print the Network Structure, number of parameters and the size of the train and test dataset

    database_size()
        Returns the size of the trainning dataset (`int`)

    parameters_count()
        Returns the number of trainnable parameters in the network (`int`)

    reset()
        Reset the parameters in the Network

    save_model(file)
        Save the model state_dict in file (`str`), preferent pth
    
    load_model(file)
        Load the model state_dict from file (`str`)

    state()
        Print the network state_dict

    show_plots()
        Show the plots builded during trainning
    
    close_plots()
        Close all the matplotlib.pyplot figures to save memory

    '''
    def __init__(self, file_name, architecture, hyperparameters, config, mode='complete', workers=0, extra_route=None):
        # main config object
        self.config = config
        
        # Backend to run in tensor cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        if config.cuda['limit_threads']:
            torch.set_num_threads(1)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device

        self.path = os.path.join('Training_results', mode) # Path to store the results
        
        if extra_route:
            self.path = os.path.join(self.path, extra_route)
            self.extra_route = extra_route

        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        
        self.mode = mode # Level of training
        self.file_name = file_name # File where you'll write the results
        self.names = [ 
            'dimension',
            'architecture',
            'parameters',
            'optimizer',
            'loss_function',
            'epochs',
            'batch_size',
            'lr',
            'training_time',
            'random_state',
            'train_loss',
            'val_loss',
            'test_loss',
            'mae_val',
            'mae_test',
            'err_val',
            'err_test',
            'r2_val',
            'r2_test',
            'acc_val',
            'acc_test',
            'outliers_count'
        ] # Column names for the results file

        # Architecture variable assignment
        self.model_name = architecture['model']
        self.num_features = architecture['num_features']
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
        self.is_jpg_failed = False

        af_valid = self.config.json['af_valid']

        # Model definition
        self.model = eval(f'{self.model_name}({self.num_features}, {self.dimension}, {self.activation_functions}, {af_valid})')

        # Build routes
        self.plots_path, self.img_path, self.pred_path = self.__build_routes()

        # Outliers value
        self.outliers_count = 0
        self.Outliers_DF = []

        # Drop molecules
        if config.configurations['drop']:
            self.drop = config.inputs['drop_file']
        else:
            self.drop = False

        # Loader Classes
        self.processer = PreprocessData(config)
        self.loader = DatabaseLoader(config)

        # ID values
        self.ID, self.x, self.y = self.processer.Retrieve_Processed()

        # smiles_database
        self.smiles_database = config.smiles_database

        # datasets
        self.random_state = config.custom['random_state']
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(config)

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

        img_path = os.path.join(plots_path, 'img_mols')

        pred_path = os.path.join(self.path, 'Predictions')

        if not os.path.isdir(img_path):
            os.makedirs(img_path)

        if not os.path.isdir(pred_path):
            os.makedirs(pred_path)

        return plots_path, img_path, pred_path

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

    def __save_checkpoint(self, epoch, optimizer, file):
        '''
        Saves a checkpoint for the current network 
        '''
        checkpoint = {
            'epoch' : epoch,
            'model_state' : self.model.state_dict(),
            'optim_state' : optimizer.state_dict()
        }

        torch.save(checkpoint, file)

    def __load_checkpoint(self, optimizer, file):
        '''
        Loads the checkpoint to current network
        '''
        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        epoch = checkpoint['epoch']

        return epoch, optimizer

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

        ax1.plot(self.values_plot['l_train'], '-r')
        ax1.plot(self.values_plot['l_val'], ':g')
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
    
    def write_metrics(self, values):
        '''
        Save all the results inside csv file
        '''
        cols = [str(val) for val in self.names]

        if not os.path.isfile(os.path.join(self.path, self.file_name)):
            with open(os.path.join(self.path, self.file_name), 'w') as txt:
                txt.write(','.join(cols))
        
        ff = [
            str(self.dimension).replace(',', '|'),
            str(self.activation_functions).replace(', ', '|'),
            self.parameters,
            self.optim,
            self.crit,
            self.num_epochs,
            self.batch_size,
            self.learning_rate,
            round(values[0], 2),
            self.random_state,
            round(values[1], 4),
            round(values[2], 4),
            values[3].round(4),
            round(values[4], 4),
            round(values[5], 4),
            round(values[6], 4),
            round(values[7], 4),
            round(values[8], 4),
            round(values[9], 4),
            round(values[10]*100, 2),
            round(values[11]*100, 2),
            self.outliers_count
            ]
        
        ff_str = [str(f) for f in ff]

        line = ','.join(ff_str)
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

        if np.isnan(np.sum(y)) or np.isnan(np.sum(y_pred)):
            return 1, 1, 0, 1, np.ones_like(y)

        else:
            lineal.fit(y, y_pred)
            r2 = lineal.score(y, y_pred)
            linear_predicted = lineal.predict(y)

            MAE = mean_absolute_error(y, y_pred)
            MSE = mean_squared_error(y, y_pred)
            RMSE = np.sqrt(MSE)

            acc = abs(1 - RMSE)

            return MAE, RMSE, acc, r2, linear_predicted

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
    
    def __save_img_outliers(self):
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit import RDLogger

        # Disable logs and errors
        RDLogger.DisableLog('rdApp.*')

        smiles_df = pd.read_csv(self.smiles_database)
        n_pics = self.config.configurations['n_pics']

        for i, ID in enumerate(self.Outliers_DF['ID'].values[0:n_pics]):
            row = smiles_df[smiles_df['ID'] == ID]
            ID_smile = row['smiles'].values[0]

            Mol = Chem.MolFromSmiles(ID_smile)

            file_name = os.path.join(self.img_path, f'{i+1}_{ID}.jpg')

            if Mol:
                img = Draw.MolToImage(Mol)
                img.save(file_name)
            else:
                self.is_jpg_failed = True
        
    def start_training(self, write=True, allow_print=False, save_plots=False, monitoring=False):
        '''
        Runs training
        '''
        # load model to device
        self.model.to(self.device)

        # Instance dataloaders
        train_loader, train_loader_full, val_loader, test_loader = self.__instance_Dataloaders()
        
        # Training full tensors
        for _, (x_tr, y_tr) in enumerate(train_loader_full):
            x_training, y_training = x_tr, y_tr

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
        l_train = []
        l_val = []
        l_acc_val = []
        r2_val = []

        # ======================================================================
        # ========================== Training loop =============================
        # ======================================================================
        st = time.time() # Start time

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):

            for _, (x_train, y_train) in enumerate(train_loader):
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
                x_train = copy.deepcopy(x_training)
                y_train = copy.deepcopy(y_training)
                
                # Load values to device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                with torch.autocast(device_type='cuda'):
                    ytrain_pred = self.model(x_train)
                    loss_train_metric = criterion(ytrain_pred, y_train)

                l_train.append(loss_train_metric.item())

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

                l_val.append(loss_val.item())

                MAE_val, RMSE_val, accVal, r2Val, _ = self.__compute_metrics(y_val, yval_pred)

                # Store metrics
                l_acc_val.append(accVal)
                r2_val.append(r2Val)

            # Check if model is stuck each 50 epochs
            if (epoch+1)%50 == 0:
                if np.mean(l_acc_val[-15::]) == 0:
                    break

            # Deep copy the model if monitoring
            if monitoring:
                if accVal > best_acc:
                    best_acc = accVal
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            if allow_print:
                if epoch == 0:
                    print('\n', '#'*37, ' Training Progress ', '#'*37, '\n')

                if (epoch+1)%10 == 0:
                    print(f'Epoch: {(epoch+1):04} Validation: MAE = {MAE_val:.4f} ERR = {RMSE_val:.4f} ACC = {accVal*100:.2f} r2 = {r2Val:.4f}', end='\r')

        # ===== Restore best when monitoring =====
        if monitoring:
            print(f'\nBetter performance: epoch = {best_epoch} ___ acc = {best_acc}')

            self.config.update(
                best_acc=best_acc,
                best_epoch=best_epoch
            )
            self.model.load_state_dict(best_model_wts)

            # Recompute steps
            with torch.no_grad():
                # -----------------------Training------------------------------
                # Unique step
                x_train = copy.deepcopy(x_training)
                y_train = copy.deepcopy(y_training)
                
                # Load values to device
                x_train = x_train.to(self.device)
                y_train = y_train.to(self.device)

                with torch.autocast(device_type='cuda'):
                    ytrain_pred = self.model(x_train)
                    loss_train_metric = criterion(ytrain_pred, y_train)

                l_train.append(loss_train_metric.item())

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

                l_val.append(loss_val.item())

                MAE_val, RMSE_val, accVal, r2Val, _ = self.__compute_metrics(y_val, yval_pred)

                # Store metrics
                l_acc_val.append(accVal)
                r2_val.append(r2Val)
    
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

        l_tt = loss_test.item()
        l_tt = torch.as_tensor(l_tt)

        MAE_test, RMSE_test, acc_test, r2_test, _ = self.__compute_metrics(y_test, ytest_pred)

        et = time.time() # End time
        elapsed_time = et - st

        self.parameters_count() # To define self.parameters

        values = [
            elapsed_time, loss_train_metric.item(), loss_val.item(), l_tt.numpy(), MAE_val, 
            MAE_test, RMSE_val, RMSE_test, r2_val[-1], r2_test, l_acc_val[-1], acc_test
        ]

        self.values_plot = {'l_train' : l_train,
                        'l_val' : l_val,
                        'acc_val' : l_acc_val,
                        'y_val' : y_val,
                        'yval_pred' : yval_pred,
                        'y_test' : y_test,
                        'ytest_pred' : ytest_pred,
                        'r2_val' : r2_val,
                        'r2_test' : r2_test
                        }

        if write:
            self.save_model(os.path.join(self.plots_path, 'model.pth'))
            self.write_predictions()
            self.write_metrics(values)
            self.write_config(os.path.join(self.plots_path, 'config.ini'))

        if save_plots:
            self.__build_plots(save=True)