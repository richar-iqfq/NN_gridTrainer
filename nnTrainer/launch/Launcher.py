import os
import time
import itertools
import copy
import logging

from tqdm import tqdm
from scipy.optimize import minimize
import numpy as np

from .. import Configurator
from nnTrainer.data.Reader import Reader
from nnTrainer.data.Recorder import Recorder
from nnTrainer.train.Trainer import Trainer
from nnTrainer.tools.Train import (
    get_adjacent_batches,
    get_random_activation_functions,
)

class Launcher():
    '''
    Customizable class to launch multiple architectures to training and watch the progress in status bar
    '''
    def __init__(self, perform: list, tol: float=0.12):
        self.Networks: dict = {
            1 : 'Net_1Hlayer',
            2 : 'Net_2Hlayer',
            3 : 'Net_3Hlayer',
            4 : 'Net_4Hlayer',
            5 : 'Net_5Hlayer',
            6 : 'Net_6Hlayer'
        }

        # Hidden size to start the searching after the grid step
        self.initial_hidden_size_after_grid: int = 1

        # config property
        self.config: Configurator = Configurator()

        # Record training
        self.record_training(perform)

        # Perform steps
        self.perform: list = perform

        # Number of parameters tolerance
        self.tol: float = tol

        # Architecture loading
        self.num_features: int = self.config.get_json('num_features')
        self.num_targets: int = self.config.get_json('num_targets')
        self.optimizers: list = self.config.get_json('optimizers')
        self.loss_functions: list = self.config.get_json('loss_functions')
        self.af_list: tuple = copy.deepcopy(self.config.get_json('af_valid'))
        self.af_list.remove('None')

        # Inputs
        self.parted: int = self.config.get_custom('parted')
        self.extra_name: str = self.config.get_custom('extra_filename')
        self.seed: int = self.config.get_custom('seed')
        self.lineal_output: bool = self.config.get_inputs('lineal_output')

        # Configurations
        self.max_hidden_layers: int = self.config.get_configurations('max_hidden_layers')
        self.min_neurons: int = self.config.get_configurations('min_neurons')
        self.max_neurons: int = self.config.get_configurations('max_neurons')
        self.lr_range: float = self.config.get_configurations('learning_rate_range')
        self.bz_range: tuple = self.config.get_configurations('batch_size_range')
        self.tries: int = self.config.get_configurations('n_tries')
        self.n_networks: int = self.config.get_configurations('n_networks')
        self.start_point: int = self.config.get_configurations('start_point')
        self.save_plots: bool = self.config.get_configurations('save_plots')
        self.reader_criteria: str = self.config.get_configurations('reader_criteria')
        self.workers: int = self.config.get_configurations('workers')
        self.train_ID: str = self.config.get_inputs('train_ID')

        logging.basicConfig(filename=f'logs/{self.train_ID}.log', level=logging.INFO)

    def launch(self, trainer: Trainer) ->tuple:
        '''
        launch the training and assure the number of parameters is on range

        Parameters
        ----------
        trainer `object of class Trainer`
        '''
        n_parameters = trainer.parameters_count()
        n_train = trainer.database_size()
        
        # flag to know if training went wrong
        flag = True
        
        if self.tol <= 0.12:
            if n_parameters <= self.tol*n_train:
                # try:
                time.sleep(1)
                trainer.start_training(save_plots=self.save_plots, monitoring=True)

                trainer.close_plots()
                # except:
                #     flag = False
        else:
            if n_parameters > 0.12*n_train and n_parameters <= self.tol*n_train:
                try:
                    time.sleep(1)
                    trainer.start_training(save_plots=self.save_plots, monitoring=True)

                    trainer.close_plots()
                except:
                    flag = False

        return flag, trainer

    def record_training(self, perform: list) -> None:
        recorder = Recorder()

        recorder.save_values(perform)

    def recover_network(self, hidden_layers: int, step: str, worst: bool=False) -> list:
        # Load better network parameters
        try:
            rd = Reader(hidden_layers, f'_{hidden_layers}Hlayer{self.extra_name}', step=step)
        except:
            return None
        
        better_network = rd.recover_best(n_values=self.n_networks, criteria=self.reader_criteria, worst=worst)
        
        return better_network

    def explore_lr(self, previous_step: str, network: dict=False) -> None:
        pass

    def grid(self, previous_step: str, network: dict=False, modes: list=['assembly', 'random']) -> None:
        actual_step = 'grid'

        logging.info('Started grid search')

        print('+'*50)
        print('Performing grid search...\n')

        #============================== Assembly ==================================================== 
        if 'assembly' in modes:
            logging.info('Started assembly mode')
            
            # Build hidden layers
            if self.lineal_output:
                initial_af = [(p, 'None') for p in self.af_list]
            else:
                initial_af = [p for p in itertools.product(self.af_list, repeat=2)]

            total_architectures = len(initial_af)*(self.max_neurons - (self.min_neurons-1) )

            # Search for better AF and Dimension over each network
            for hidden_size in range(self.start_point, self.max_hidden_layers+1):
                logging.info(f'Searching {hidden_size} layers')

                Network = self.Networks[hidden_size]
                if hidden_size != 1:
                    total_architectures = (self.max_neurons - (self.min_neurons-1) )*len(self.af_list)

                print('.'*50)
                print(Network)
                print('.'*50, '\n')

                file = Network + f'{self.extra_name}.csv'
                last_hidden_size = hidden_size-1
                
                pbar = tqdm(total=total_architectures, desc='Architectures', colour='green')

                for dim in range(self.min_neurons, self.max_neurons+1):
                    
                    if hidden_size == 1:
                        dimension = (dim, )

                        for af in initial_af:

                            architecture = {
                                'model' : Network,
                                'num_targets' : self.num_targets,
                                'num_features' : self.num_features,
                                'dimension' : dimension,
                                'activation_functions' : af,
                                'optimizer' : self.config.get_loss('optimizer'),
                                'criterion' : self.config.get_loss('criterion')
                            }

                            tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                            train_flag, tr = self.launch(tr)

                            if not train_flag:
                                pbar.update()
                                continue

                            pbar.update()

                    else:
                        better_network = self.recover_network(last_hidden_size, step=actual_step)

                        if better_network == None:
                            logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                            print(f'Any functional model found for {hidden_size} hidden layers...')
                            continue
                        
                        for i, network_step in enumerate(better_network):
                            dimension = list(network_step['dimension'])
                            dimension.append(dim)
                            dimension = tuple(dimension)
                            
                            initial_af = list(network_step['activation_functions'])
                            
                            for af in self.af_list:
                                
                                final_af = copy.copy(initial_af)
                                    
                                if self.lineal_output:
                                    final_af.insert(-1, af)
                                else:
                                    final_af.append(af)
                        
                                final_af = tuple(final_af)
                                
                                architecture = {
                                    'model' : Network,
                                    'num_targets' : self.num_targets,
                                    'num_features' : self.num_features,
                                    'dimension' : dimension,
                                    'activation_functions' : final_af,
                                    'optimizer' : self.config.get_loss('optimizer'),
                                    'criterion' : self.config.get_loss('criterion')
                                }

                                tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                                train_flag, tr = self.launch(tr)

                                if not train_flag:
                                    pbar.update()
                                    continue

                                pbar.update()

        if 'random' in modes:
            logging.info('Started ramdom mode')

            for hidden_size in range(self.start_point, self.max_hidden_layers+1):

                if hidden_size > 1:
                    logging.info(f'Searching {hidden_size} layers')
                    Network = self.Networks[hidden_size]
                    file = Network + f'{self.extra_name}.csv'

                    print('.'*50)
                    print(Network)
                    print('.'*50, '\n')

                    better_network = self.recover_network(hidden_size, step=actual_step)

                    if better_network == None:
                        logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                        print(f'Any functional model found for {hidden_size} hidden layers...')
                        continue
                    
                    for i, network_step in enumerate(better_network):
                        final_af_list = get_random_activation_functions(self.af_list, hidden_size+1, self.lineal_output)
                        
                        total_architectures = len(final_af_list)

                        pbar = tqdm(total=total_architectures, desc='Architectures', colour='green')

                        for final_af in final_af_list:
                            architecture = {
                                'model' : Network,
                                'num_targets' : self.num_targets,
                                'num_features' : self.num_features,
                                'dimension' : network_step['dimension'],
                                'activation_functions' : final_af,
                                'optimizer' : self.config.get_loss('optimizer'),
                                'criterion' : self.config.get_loss('criterion')
                            }

                            tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                            train_flag, tr = self.launch(tr)

                            if not train_flag:
                                pbar.update()
                                continue
                            
                            pbar.update()

        logging.info('Grid search complete')
        print('\nGrid search complete...\n')
        print('+'*50)

    def optimization(self, previous_step: str, network: dict=False) -> None:
        actual_step = 'optimization'

        logging.info('Started optimization search')

        print('\n', '+'*50)
        print('Performing optimization...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers')
            Network = self.Networks[hidden_size]

            print('.'*50)
            print(Network)
            print('.'*50, '\n')
            
            # Search for better optimizer and criterion over network
            if network:
                print('Running specific network\n')
                better_network = network
            else:
                better_network = self.recover_network(hidden_size, step=previous_step)

                if better_network == None:
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue
        
            for i, network_step in enumerate(better_network):
                
                if self.n_networks > 1:
                    print(f'Runing Network Test {i+1}/{self.n_networks}\n')

                pbar = tqdm(total=len(self.optimizers)*len(self.loss_functions), desc='Optimizers', colour='green')

                file = Network + f'{self.extra_name}.csv'

                for optimizer in self.optimizers:

                    for criterion in self.loss_functions:

                        architecture = {
                            'model' : Network,
                            'num_targets' : self.num_targets,
                            'num_features' : self.num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : network_step['activation_functions'],
                            'optimizer' : optimizer,
                            'criterion' : criterion
                        }

                        tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                        train_flag, tr = self.launch(tr)

                        if not train_flag:
                            pbar.update()
                            continue

                        pbar.update()
        
        logging.info('Optimization search complete')
        print('\nOptimization search complete...\n')
        print('+'*50)

    def restart_grid_from_worst(self, previous_step: str, network: dict=False) -> None:
        actual_step = 'restart_worst'
        logging.info('Restarted from worst optimizer')
        print('\n', '+'*50)
        print('Performing grid from worst...\n')
    
        worst_network = self.recover_network(4, previous_step, worst=True)

        if worst_network == None:
            logging.info(f"Any functional model found for {4} hidden layers in {actual_step}")
            print(f'Any functional model found for {4} hidden layers...')

        self.config.update(
            optimizer = worst_network[0]['optimizer'],
            criterion = worst_network[0]['criterion']
        )

        self.grid('', modes=['assembly', ''])

    def random_state(self, previous_step: str, network: dict=False) -> None:
        actual_step = 'random_state'
        logging.info('Random state search started')

        print('\n', '+'*50)
        print('Performing random_state searching...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers')

            Network = self.Networks[hidden_size]

            print('.'*50)
            print(f'{Network} Parted: {self.parted}')
            print('.'*50, '\n')

            # Search for better learning_rate in network
            if network:
                print('Running specific network\n')
                better_network = network
            else:
                better_network = self.recover_network(hidden_size, step=previous_step)

                if better_network == None:
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue
            
            rnd = np.random.RandomState(seed=self.seed)
            random_state_list = rnd.randint(150, 200000, self.tries)

            file = Network + f'{self.extra_name}_RS'

            par = len(random_state_list) // 3

            if self.parted == None:
                random_states = random_state_list
            elif self.parted == 1:
                random_states = random_state_list[0:par]
                file += '1'
            elif self.parted == 2:
                random_states = random_state_list[par:2*par]
                file += '2'
            else:
                random_states = random_state_list[2*par::]
                file += '3'

            file += '.csv'

            n_iterations = len(random_states)
            
            for i, network_step in enumerate(better_network):
                
                if self.n_networks > 1:
                    print(f'Runing Network Test {i+1}/{self.n_networks}\n')
                
                pbar = tqdm(total=n_iterations, desc='Random States', colour='green')

                for RS in random_states:

                    architecture = {
                        'model' : Network,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    self.config.update(
                        batch_size = network_step['batch_size'],
                        learning_rate = round(float(network_step['lr']), 9),
                        random_state = RS
                    )

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)
                    
                    if not train_flag:
                        pbar.update()
                        continue

                    pbar.update()

        logging.info('Random state search complete')
        print('\nrandom_state search complete...\n')
        print('+'*50)

    def tuning_batch(self, previous_step: str, network: dict=False) -> None:
        actual_step = 'tuning_batch'

        logging.info(f'Tuning batch search started (Parted: {self.parted})')
        print('\n', '+'*50)
        print('Performing tuning batch search...\n')
        
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers (Parted: {self.parted})')
            Network = self.Networks[hidden_size]

            print('.'*50)
            print(f'{Network} Parted: {self.parted}')
            print('.'*50, '\n')
                
            # Search for better batch size in network
            if network:
                print('Running specific network\n')
                better_network = network
            else:
                better_network = self.recover_network(hidden_size, step=previous_step)

                if better_network == None:
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue

            file = Network + f'{self.extra_name}_batches'
            rnd = np.random.RandomState(seed=self.seed)

            mini_batches = rnd.randint(self.bz_range[0], self.bz_range[1], self.tries)

            par = len(mini_batches) // 3

            if self.parted == None:
                batches = mini_batches
            elif self.parted == 1:
                batches = mini_batches[0:par]
                file += '1'
            elif self.parted == 2:
                batches = mini_batches[par:2*par]
                file += '2'
            else:
                batches = mini_batches[2*par::]
                file += '3'

            file += '.csv'

            n_iterations = len(batches)
            
            for i, network_step in enumerate(better_network):

                if self.n_networks > 1:
                    print(f'Runing Network Test {i+1}/{self.n_networks}\n')

                pbar = tqdm(total=n_iterations, desc='Batches', colour='green')

                for batch_size in batches:

                    architecture = {
                        'model' : Network,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    self.config.update(batch_size=int(batch_size))

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)

                    if not train_flag:
                        pbar.update()
                        continue
                        
                    pbar.update()

            ######################## Testing nearest powers of two in batches ##########################
            logging.info(f'Searching nearest powers of two in batches (Parted: {self.parted})')
            print('\n', '#'*16, ' Batch powering... ', '#'*16, '\n')
            
            better_network = self.recover_network(hidden_size, step=actual_step)

            if better_network == None:
                logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                print(f'Any functional model found for {hidden_size} hidden layers...')
                continue
            
            for i, network_step in enumerate(better_network):
                new_batches = get_adjacent_batches(network_step['batch_size'])

                for batch_size in new_batches:

                    print(f'Batch size => {batch_size}\n')

                    architecture = {
                        'model' : Network,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    self.config.update(batch_size=int(batch_size))

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)

                    if not train_flag:
                        pbar.update()
                        continue

            ######################## Testing three increments and decrements in batches ##########################
            logging.info(f'Searching increments in batches (Parted: {self.parted})')
            print('\n', '#'*16, ' Testing three increments... ', '#'*16, '\n')
            
            better_network = self.recover_network(hidden_size, step=actual_step)

            if better_network == None:
                logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                print(f'Any functional model found for {hidden_size} hidden layers...')
                continue
            
            for i, network_step in enumerate(better_network):
                new_batches = []

                for j in range(1, 4):
                    new_batches.append(network_step['batch_size']+j)
                    new_batches.append(network_step['batch_size']-j)

                for batch_size in new_batches:

                    print(f'Batch size => {batch_size}\n')

                    architecture = {
                        'model' : Network,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    self.config.update(batch_size=int(batch_size))

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)

                    if not train_flag:
                        pbar.update()
                        continue
        
        logging.info(f'Tuning batch search complete (Parted: {self.parted})')
        print('\nBatch search complete...\n')
        print('+'*50)

    def tuning_lr(self, previous_step: str, network: dict=False) -> None:
        actual_step = 'tuning_lr'

        logging.info(f'Learning rate search started (Parted: {self.parted})')

        print('\n', '+'*50)
        print('Performing learning rate search...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers (Parted: {self.parted})')

            Network = self.Networks[hidden_size]

            print('.'*50)
            print(f'{Network} Parted: {self.parted}')
            print('.'*50, '\n')

            # Search for better learning_rate in network
            if network:
                print('Running specific network\n')
                better_network = network
            else:
                better_network = self.recover_network(hidden_size, step=previous_step)

                if better_network == None:
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue
            
            file = Network + f'{self.extra_name}_lr'
            rnd = np.random.RandomState(seed=self.seed)

            lr_list = rnd.uniform(self.lr_range[0], self.lr_range[1], self.tries)
            
            par = len(lr_list) // 3

            if self.parted == None:
                learning_rates = lr_list
            elif self.parted == 1:
                learning_rates = lr_list[0:par]
                file += '1'
            elif self.parted == 2:
                learning_rates = lr_list[par:2*par]
                file += '2'
            else:
                learning_rates = lr_list[2*par::]
                file += '3'

            file += '.csv'

            n_iterations = len(learning_rates)

            for i, network_step in enumerate(better_network):
                
                if self.n_networks > 1:
                    print(f'Runing Network Test {i+1}/{self.n_networks}\n')

                pbar = tqdm(total=n_iterations, desc='learning rates', colour='green')

                for learning_rate in learning_rates:

                    architecture = {
                        'model' : Network,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    self.config.update(
                        batch_size=network_step['batch_size'],
                        learning_rate=round(learning_rate, 9)
                        )
                    
                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)

                    if not train_flag:
                        pbar.update()
                        continue

                    pbar.update()
        
        logging.info(f'Learning rate search complete (Parted: {self.parted})')
        print('\nLearning rate search complete...\n')
        print('+'*50)

    def lineal(self, previous_step: str, network: dict=False) -> None:
        actual_step = 'lineal'

        logging.info('Lineal search started')

        print('\n', '+'*50)
        print('Performing linear searching...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers')
            Network = self.Networks[hidden_size]

            print('.'*50)
            print(Network)
            print('.'*50, '\n')

            # Search for better learning_rate in network
            if network:
                print('Running specific network\n')
                better_network = network
            else:
                better_network = self.recover_network(hidden_size, step=previous_step)

                if better_network == None:
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue

            file = Network + f'{self.extra_name}_l' #!!!!!!!!!!!!!!!!!!!
            file += '.csv'

            for i, network_step in enumerate(better_network):
                
                if self.n_networks > 1:
                    print(f'Runing Network Test {i+1}/{self.n_networks}\n')

                def model_function(lr):
                    architecture = {
                        'model' : Network,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    self.config.update(
                        batch_size=network_step['batch_size'],
                        learning_rate=round(lr[-1], 9)
                    )

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)

                    if train_flag:
                        if hasattr(tr, 'values_plot'):
                            if 'acc' in self.reader_criteria:
                                err = abs(1 - tr.values_plot['acc_val'][-1])
                            else:
                                err = abs(1 - tr.values_plot['r2_val'][-1])

                            print(f'{Network}: {lr[-1]} -> {err:.4f}')
                        else:
                            err = 1
                            print(f'{Network}: No values_plot defined')

                    else:
                        print('Error in training...')
                        err = 1

                    return err

                def run():
                    lr_local = network_step['lr']

                    if 'acc' in self.reader_criteria:
                        err_0 = abs(1 - network_step['acc']/100)
                    else:
                        err_0 = abs(1 - network_step['r2'])
                    
                    print(f"{Network}: {lr_local} -> {err_0:.4f}")

                    result = minimize(model_function, lr_local, method='Nelder-Mead')
                    return result
                
                result_lr = run()
                better_lr = result_lr.x[0]
                
                print(f'\n Better lr -> {better_lr:.4f}')

        logging.info('Lineal search complete')
        print('\nLinear search complete...\n')
        print('+'*50)

    def around_exploration(self, previous_step: str, network: dict=False) -> None:
        actual_step = 'around_exploration'

        logging.info('Around exploration started')

        print('\n', '+'*50)
        print('Performing around exploration ...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers')
            Network = self.Networks[hidden_size]

            print('.'*50)
            print(f'{Network} Parted: {self.parted}')
            print('.'*50, '\n')

            # Search for better learning_rate in network
            if network:
                print('Running specific network\n')
                better_network = network
            else:
                better_network = self.recover_network(hidden_size, step=previous_step)

                if better_network == None:
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue

            file = Network + f'{self.extra_name}_RE'

            for i, network_step in enumerate(better_network):
                
                if self.n_networks > 1:
                    print(f'Runing Network Test {i+1}/{self.n_networks}\n')

                initial_lr = float(network_step['lr'])
                lr_list = []
 
                initial_batch = network_step['batch_size']
                if initial_batch == 'All':
                    initial_batch = 200

                batch_list = []

                rs = int(network_step['random_state'])
                n_lr = 0
                n_batch = 0

                for i in range(1, 16):
                    if initial_lr + i*0.1 < 1:
                        lr_list.append(initial_lr + i*0.1)
                        n_lr += 1
                    if initial_lr - i*1E-4 > 0:
                        lr_list.append(initial_lr - i*1E-4)
                        n_lr += 1

                    if i<=6:
                        batch_list.append(initial_batch + i*1)
                        batch_list.append(initial_batch - i*1)
                        n_batch += 2
                
                par = len(lr_list) // 3
                
                if self.parted == None:
                    lr_selected = lr_list
                elif self.parted == 1:
                    lr_selected = lr_list[0:par]
                    file += '1'
                elif self.parted == 2:
                    lr_selected = lr_list[par:2*par]
                    file += '2'
                else:
                    lr_selected = lr_list[2*par::]
                    file += '3'

                file += '.csv'
                
                n_values = len(lr_selected)*len(batch_list)
                pbar = tqdm(total=n_values, desc='Exploration steps', colour='green')

                for lr in lr_selected:
                    architecture = {
                        'model' : Network,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    for batch in batch_list:
                        self.config.update(
                            batch_size = batch,
                            learning_rate = round(lr, 9),
                            random_state = rs
                        )

                        tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=actual_step, workers=self.workers)

                        train_flag, tr = self.launch(tr)

                        if not train_flag:
                            pbar.update()
                            continue

                        pbar.update()

            logging.info('Around exploration complete')
            print('\naround_exploration search complete...\n')
            print('+'*50)

    def Test_model(self, network: dict, overview: bool=True, monitoring: bool=False) -> None:
        '''
        Run a specific model and save it to folder Models/recovering

        Parameters
        ----------
        network : object of class Reader().recover_best()
            Dictionary with the parameters for the model to save
        
        overview : `bool`
            If True will print on console the full network information

        monitoring : `bool`
            If True the file will keep the better result obtained during training
        '''
        # Set random state from configurator class
        self.config.update(
            random_state = network['random_state']
        )

        # Set architecture and hyperparameters
        hidden_layers = network['hidden_layers']
        Network = self.Networks[hidden_layers]

        architecture = {
            'model' : Network,
            'num_targets' : self.num_targets,
            'num_features' : self.config.get_json('num_features'),
            'dimension' : network['dimension'],
            'activation_functions' : network['activation_functions'],
            'optimizer' : network['optimizer'],
            'criterion' : network['criterion']
        }

        Hyperparameters = {
            'num_epochs' : self.config.get_hyperparameters('num_epochs'),
            'batch_size' : int(network['batch_size']),
            'learning_rate' : round(network['lr'], 9)
        }

        extra_name = self.config.get_custom('extra_filename')
        save_plots = self.config.get_configurations('save_plots')

        file = Network + f'{extra_name}.csv'
        tr = Trainer(file, architecture, Hyperparameters, step='recovering')
        
        if overview:
            tr.overview()

        tr.start_training(write=True , allow_print=True, save_plots=save_plots, monitoring=monitoring)
    
    def Run(self, network: dict=False, last_step: str=False) -> None:
        '''
        Start the analysis

        Parameters
        ----------
        network : `dict`
            Dictionary from the ResultsReader.recover_best()
        
        last_step : `str` (optional)
            Last performed step, default is False
        '''
        os.system('clear')
        np.seterr(all="ignore")

        step_functions = {
            'grid' : self.grid,
            'optimization' : self.optimization,
            'random_state' : self.random_state,
            'tuning_batch' : self.tuning_batch,
            'tuning_lr' : self.tuning_lr,
            'lineal' : self.lineal,
            'around_exploration' : self.around_exploration,
            'restart_grid_from_worst' : self.restart_grid_from_worst
        }

        for i, step in enumerate(self.perform):
            if step == 'grid':
                previous_step = last_step
            
            else:
                if 'worst' in step:
                    previous_step = 'grid'

                elif step == 'optimization':
                    if last_step:
                        previous_step = last_step
                    else:
                        previous_step = 'grid'

                else:
                    if i-1 >= 0:
                        previous_step = self.perform[i-1]
                    else:
                        previous_step = last_step

            # Define step function
            launcher_function = step_functions[step]

            # Execute function
            launcher_function(previous_step, network)