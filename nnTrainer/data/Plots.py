import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from .. import Configurator
from nnTrainer.data.Metrics import (
    OutliersComputer,
    MetricsComputer
)
from nnTrainer.data.Preprocess import PreprocessData
from nnTrainer.tools.Array import process_tensor

class PlotsBuilder():
    def __init__(self, plots_path: str) -> None:
        self.config = Configurator()
        self.save_plots = self.config.get_configurations('save_plots')
        self.plots_path = plots_path
        self.targets = self.config.get_json('targets')

        self.processer = PreprocessData()
        self.metrics_calc = MetricsComputer()
        self.outliers_calc = OutliersComputer()

        self.dpi = 450 # dpi
        self.w = 20
        self.h = 13

    def __make_regression_path(self):
        '''
        Build regression folder
        '''
        fig_regression_path = os.path.join(self.plots_path, 'sets_regression')

        if self.save_plots:
            if not os.path.isdir(fig_regression_path):
                os.makedirs(fig_regression_path)

        return fig_regression_path
    
    def __make_full_regression_path(self):
        fig_full_regression_path = os.path.join(self.plots_path, 'full_regression')

        if self.save_plots:
            if not os.path.isdir(fig_full_regression_path):
                os.makedirs(fig_full_regression_path)

        return fig_full_regression_path
    
    def __make_full_unscaled_path(self):
        fig_full_unscaled_path = os.path.join(self.plots_path, 'full_unscalled_regression')

        if self.save_plots:
            if not os.path.isdir(fig_full_unscaled_path):
                os.makedirs(fig_full_unscaled_path)

        return fig_full_unscaled_path

    def loss_plot(self, values_plot: dict):
        figure, axis = plt.subplots(1)
        figure.set_size_inches(self.w, self.h)

        axis.plot(values_plot['loss_train_list'], '-r')
        axis.plot(values_plot['loss_validation_list'], ':g')
        axis.legend(['Train', 'Validation'])
        axis.set_title('Loss', weight='bold')
        axis.set_xlabel('Epochs')
        axis.set_ylabel('Loss')

        if self.save_plots:
            figure.savefig(os.path.join(self.plots_path, 'loss.pdf'), dpi=self.dpi, format='pdf')
    
    def regression_plots(self, values_plot: dict):
        fig_regression_path = self.__make_regression_path()

        y_train = values_plot['y_train']
        ytrain_pred = values_plot['ytrain_pred']
        
        y_val = values_plot['y_val']
        yval_pred = values_plot['yval_pred']

        y_test = values_plot['y_test']
        ytest_pred = values_plot['ytest_pred']

        # Detach and move to cpu
        y_train = process_tensor(y_train)
        ytrain_pred = process_tensor(ytrain_pred)

        y_val = process_tensor(y_val)
        yval_pred = process_tensor(yval_pred)
        
        y_test = process_tensor(y_test)
        ytest_pred = process_tensor(ytest_pred)


        r2_train = values_plot['r2_train']
        r2_val = values_plot['r2_val']
        r2_test = values_plot['r2_test']

        boolean_outliers_train, _ = self.outliers_calc.get_numerical_outliers(y_train, ytrain_pred)
        boolean_outliers_val, _ = self.outliers_calc.get_numerical_outliers(y_val, yval_pred)
        boolean_outliers_test, _ = self.outliers_calc.get_numerical_outliers(y_test, ytest_pred)

        # Build and save plots
        for i, target in enumerate(self.targets):
            fig_regression, ax_regression = plt.subplots(3)
            plt.subplots_adjust(
                top=0.94,
                bottom=0.06,
                left=0.09,
                right=0.96,
                hspace=0.24,
            )

            fig_regression.suptitle(f'Regression {target}', weight='bold')
            fig_regression.set_size_inches(self.h-3, self.w)

            y = [y_train[:,i], y_val[:,i], y_test[:,i]]
            y_pred = [ytrain_pred[:,i], yval_pred[:,i], ytest_pred[:,i]]
            
            r2 = [r2_train[target], r2_val[target], r2_test[target]]

            label = ['Train', 'Validation', 'Test']
            style = ['.g', '.r', '.m']

            boolean_outliers = [boolean_outliers_train[target], boolean_outliers_val[target], boolean_outliers_test[target]]

            for i in range(3):
                outliers_count = np.count_nonzero(boolean_outliers[i])

                ax_regression[i].plot(y[i], y[i], '-b')
                ax_regression[i].scatter(y[i][boolean_outliers[i]], y_pred[i][boolean_outliers[i]], color='yellowgreen')
                ax_regression[i].plot(y[i], y_pred[i], style[i])
                
                ax_regression[i].set_xlabel('y')
                ax_regression[i].set_ylabel('y_pred')

                ax_regression[i].set_title(f'{label[i]} r2 = {r2[i]:.4f}      outliers count = {outliers_count}', weight='bold')

            if self.save_plots:
                fig_regression.savefig(os.path.join(fig_regression_path, f"{target.replace('/', '-')}.pdf"), dpi=self.dpi, format='pdf')

    def accuracy_plot(self, values_plot: dict):
        fig_accuracy, ax_accuracy = plt.subplots(1)
        fig_accuracy.set_size_inches(self.w, self.h)

        ax_accuracy.plot(values_plot['general_acc_train'], '--r')
        ax_accuracy.plot(values_plot['general_acc_val'], '--g')

        ax_accuracy.legend(['Train', 'Validation'])

        ax_accuracy.set_xlabel('Epochs')
        ax_accuracy.set_ylabel('Accuracy %')

        ax_accuracy.set_title('General Accuracy', weight='bold')

        if self.save_plots:
            fig_accuracy.savefig(os.path.join(self.plots_path, 'acc.pdf'), dpi=self.dpi, format='pdf')

    def regression_over_epoch_plot(self, values_plot: dict):
        fig, axis = plt.subplots(1)
        fig.set_size_inches(self.w, self.h)

        axis.plot(values_plot['general_r2_train'])
        axis.plot(values_plot['general_r2_val'])

        axis.legend(['Train', 'Validation'])

        axis.set_xlabel('Epochs')
        axis.set_ylabel('r2')

        axis.set_title('General Regression coeficient', weight='bold')

        if self.save_plots:
            fig.savefig(os.path.join(self.plots_path, 'reg.pdf'), dpi=self.dpi, format='pdf')

    def full_scaled_plot(self, y: np.ndarray, y_pred: np.ndarray):
        fig_full_regression_path = self.__make_full_regression_path()

        boolean_outliers, _ = self.outliers_calc.get_numerical_outliers(y, y_pred)
        
        MAE, RMSE, acc, r2 = self.metrics_calc.compute(y, y_pred)

        outliers_general = 0

        outliers_dict = {}

        for i, target in enumerate(self.targets):
            # Variable assingment
            y_target = y[:,i]
            ytarget_pred = y_pred[:,i]
            
            # Regression plot
            fig_full, ax = plt.subplots(1)
            fig_full.suptitle(f'Full Regression {target}', weight='bold')
            fig_full.set_size_inches(self.w, self.h)
            
            ax.plot(y_target, y_target, '-b')
            ax.scatter(y_target[boolean_outliers[target]], ytarget_pred[boolean_outliers[target]], color='yellowgreen')
            ax.plot(y_target, ytarget_pred, '.r')
            
            ax.set_xlabel('y')
            ax.set_ylabel('y_pred')

            outliers_count = np.count_nonzero(boolean_outliers[target])
            outliers_dict[f'outliers_{target}'] = outliers_count
            outliers_general += outliers_count

            ax.set_title(f'r2 = {r2[target]:.4f}      outliers count = {outliers_count}', weight='bold')

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

            if self.save_plots:
                if self.config.get_custom('lineal_output'):
                    fig_full.savefig(os.path.join(fig_full_regression_path, f"{target.replace('/', '-')}_full.pdf"), dpi=self.dpi, format='pdf')
        
        # update general outliers
        outliers_dict['outliers_general'] = outliers_general

        return outliers_dict

    def full_unscaled_plot(self, y:np.ndarray, y_pred: np.ndarray):
        if self.config.get_custom('lineal_output') == False:
            y_unscaled = self.processer.y_unscale_routine(y)
            y_unscaled_pred = self.processer.y_unscale_routine(y_pred)

            boolean_outliers, _ = self.outliers_calc.get_numerical_outliers(y_unscaled, y_unscaled_pred)
                
            MAE, RMSE, acc, r2 = self.metrics_calc.compute(y_unscaled, y_unscaled_pred)

            fig_full_unscaled_path = self.__make_full_unscaled_path()

            for i, target in enumerate(self.targets):
                # variable assingment
                ytarget_unscaled = y_unscaled[:,i]
                ytarget_unscaled_pred = y_unscaled_pred[:,i]

                # Regression plot
                fig_full_unscaled, ax_unscaled = plt.subplots(1)
                fig_full_unscaled.suptitle(f'Full {target} Regression Unscaled', weight='bold')
                fig_full_unscaled.set_size_inches(self.w, self.h)
                
                ax_unscaled.plot(ytarget_unscaled, ytarget_unscaled, '-b')
                ax_unscaled.scatter(ytarget_unscaled[boolean_outliers[target]], ytarget_unscaled_pred[boolean_outliers[target]], color='yellowgreen')
                ax_unscaled.plot(ytarget_unscaled, ytarget_unscaled_pred, '.r')
                
                ax_unscaled.set_xlabel('y')
                ax_unscaled.set_ylabel('y_pred')

                ax_unscaled.set_title(f'r2 = {r2[target]:.4f}      outliers count = {np.count_nonzero(boolean_outliers[target]) }', weight='bold')

                textstr = '\n'.join([
                    f'MAE = {MAE[target]:.4f}',
                    f'RMSE = {RMSE[target]:.4f}',
                    f'acc = {acc[target]*100:.2f}'
                ])
                
                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                
                # place a text box in upper left in axes coords
                ax_unscaled.text(0.05, 0.95, textstr, transform=ax_unscaled.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
                
                if self.save_plots:
                    fig_full_unscaled.savefig(os.path.join(fig_full_unscaled_path, f"{target.replace('/', '-')}_full_unscaled.pdf"), dpi=self.dpi, format='pdf')

    def show_plots(self):
        '''
        Show the final plots of the training
        '''        
        plt.show()

    def close_plots(self):
        '''
        Close the active plots
        '''
        plt.close('all')

    def build_plots(self, values_plot: dict):
        '''
        Build all the required plots
        '''
        # Loss graphic
        self.loss_plot(values_plot)

        # Regression plot
        self.regression_plots(values_plot)
        
        # accuracy over epochs plots
        self.accuracy_plot(values_plot)
        
        # Regression over epochs
        self.regression_over_epoch_plot(values_plot)

    def build_full_plots(self, y: torch.Tensor, y_pred: torch.Tensor):
        y = process_tensor(y)
        y_pred = process_tensor(y_pred)

        # ======= Scaled Plot ===========
        outliers = self.full_scaled_plot(y, y_pred)

        # ======= Unscaled Plot =========
        self.full_unscaled_plot(y, y_pred)

        return outliers