from modules.ResultsReader import Reader
import os
from modules.Trainer import Trainer
import time
from tqdm import tqdm
import torch.nn as nn
from scipy.optimize import minimize
import itertools
import numpy as np
import copy

class Launcher():
    '''
    Class to launch multiple architectures to training and watch the progress in
    status bar

    Methods
    -------
    Run_complete(Hidden_Layers, Hyperparamenters, options, custom=None, perform=['grid', 'optimization', 'tuning_batch', 'tuning_lr'])
        Start a complete analysis over all configurations of 5HL in an enssamble mode.
    '''
    def __init__(self):
        self.Networks = {
            1 : 'Net_1Hlayer',
            2 : 'Net_2Hlayer',
            3 : 'Net_3Hlayer',
            4 : 'Net_4Hlayer',
            5 : 'Net_5Hlayer',
            6 : 'Net_6Hlayer'
        }

    def get_batches(self, batch_size):
        values = [16, 32, 64, 128, 256, 512, 1024, 2048]

        diff = 2048
        k = 0

        for i in range(len(values) - 1):
            j = i + 1

            compute = abs(batch_size - values[i]) + abs(batch_size - values[j])

            if compute <= diff:
                diff = compute
                k = i

        return values[k], values[k+1]

    def Test_model(self, network, config, overview=True, monitoring=False):
        '''
        Run an specific model and save it to folder Models/recovering

        Parameters
        ----------
        network : object of class Reader().recover_best()
            Dictionary with the parameters for the model to save

        config : Instance of Configurator `class`
            Object of class Configurator with all the information about the actual
            training launched.

        overview : `bool`
            If True will print on console the full network information

        monitoring : `bool`
            If True the file will keep the better result obtained during training
        '''
        # Set random state from configurator class
        config.update(
            random_state = network['random_state']
        )

        # Set architecture and hyperparameters
        hidden_layers = network['hidden_layers']
        Network = self.Networks[hidden_layers]

        architecture = {
            'model' : Network,
            'num_features' : config.json['num_features'],
            'dimension' : network['dimension'],
            'activation_functions' : network['activation_functions'],
            'optimizer' : network['optimizer'],
            'criterion' : network['criterion']
        }

        Hyperparameters = {
            'num_epochs' : config.hyperparameters['num_epochs'],
            'batch_size' : int(network['batch_size']),
            'learning_rate' : round(network['lr'], 9)
        }

        extra_name = config.custom['extra_filename']
        save_plots = config.configurations['save_plots']

        file = Network + f'{extra_name}.csv'
        tr = Trainer(file, architecture, Hyperparameters, config, mode='models', step='recovering')
        
        if overview:
            tr.overview()

        tr.start_training(save_plots=save_plots, allow_print=True, monitoring=monitoring)

    def Run_training(self, config, network=None, perform=['grid', 'optimization', 'tuning_batch', 'tuning_lr', 'lineal', 'random_state', 'around_exploration']):
        '''
        Start the grid training

        Parameters
        ----------
        config : `dict`
            Object of class Configurator

        network : `dict`
            Dictionary from the ResultsReader.recover_best(), when using this feature you can only train one step
            
        perform : `list`
            List with the steps to perform: grid, optimization, tuning_batch, tuning_lr, random_state,
            around_exploration
        '''
        if network:
            if len(perform) > 1:
                print(f'{len(perform)} steps in perform')
                print('When passing network parameter to trainer, can not define more than one step')
            if 'grid' in perform:
                print('Grid step not available when passing network')

        num_targets = config.json['num_targets']
        num_features = config.json['num_features']
        optimizers = config.json['optimizers']
        loss_functions = config.json['loss_functions']
        #======================================== COMPLETE ==========================================
        mode = 'complete'
        
        extra_name = config.custom['extra_filename']
        seed = config.custom['seed']
        lineal_output = config.inputs['lineal_output']

        max_hidden_layers = config.configurations['max_hidden_layers']
        min_neurons = config.configurations['min_neurons']
        max_neurons = config.configurations['max_neurons']

        lr_range = config.configurations['learning_rate_range']
        bz_range = config.configurations['batch_size_range']
        
        tries = config.configurations['n_tries']
        n_networks = config.configurations['n_networks']
        start_point = config.configurations['start_point']
        save_plots = config.configurations['save_plots']
        reader_criteria = config.configurations['reader_criteria']
        workers = config.configurations['workers']
        af_list = copy.deepcopy(config.json['af_valid'])

        af_list.remove('None')

        os.system('clear')
        print(f'Working with: {extra_name}\n')

        if 'grid' in perform:

            if lineal_output:
                initial_af = [(p, 'None') for p in af_list]
            else:
                initial_af = [p for p in itertools.product(af_list, repeat=2)]

            total_architectures = len(initial_af)*(max_neurons - (min_neurons-1) )
           
            print('+'*50)
            print('Performing grid search...\n')

            # Search for better AF and Dimension over each network
            for hidden_size in range(start_point, max_hidden_layers+1):

                Network = self.Networks[hidden_size]
                if hidden_size != 1:
                    total_architectures = (max_neurons - (min_neurons-1) )*len(af_list)

                print('.'*50)
                print(Network)
                print('.'*50, '\n')

                file = Network + f'{extra_name}.csv'
                last_hidden_size = hidden_size-1
                
                pbar = tqdm(total=total_architectures, desc='Architectures', colour='green')

                for dim in range(min_neurons, max_neurons+1):
                    
                    if hidden_size == 1:
                        dimension = (dim, )

                        for af in initial_af:

                            architecture = {
                                'model' : Network,
                                'num_targets' : num_targets,
                                'num_features' : num_features,
                                'dimension' : dimension,
                                'activation_functions' : af,
                                'optimizer' : config.loss['optimizer'],
                                'criterion' : config.loss['criterion']
                            }

                            tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='grid', workers=workers)

                            n_parameters = tr.parameters_count()
                            n_train = tr.database_size()

                            if n_parameters <= 0.12*n_train:
                                time.sleep(1)
                                try:
                                    tr.start_training(save_plots=save_plots)

                                    tr.close_plots()
                                    del(tr)
                                except:
                                    pbar.update()
                                    continue

                            pbar.update()

                    else:
                        try:
                            rd = Reader(last_hidden_size, f'_{last_hidden_size}Hlayer{extra_name}.csv', type='complete', step='grid')
                            better_network = rd.recover_best(criteria=reader_criteria)
                        except:
                            print(f'No files found for {Network}')
                            continue

                        if better_network == None:
                            print('Any functional model found for {}...')
                            break
                        else:
                            better_network = better_network[0]

                        dimension = list(better_network['dimension'])
                        dimension.append(dim)
                        dimension = tuple(dimension)
                        
                        initial_af = list(better_network['activation_functions'])
                        
                        for af in af_list:
                            
                            final_af = copy.copy(initial_af)
                                
                            if lineal_output:
                                final_af.insert(-1, af)
                            else:
                                final_af.append(af)
                    
                            final_af = tuple(final_af)
                            
                            architecture = {
                                'model' : Network,
                                'num_targets' : num_targets,
                                'num_features' : num_features,
                                'dimension' : dimension,
                                'activation_functions' : final_af,
                                'optimizer' : config.loss['optimizer'],
                                'criterion' : config.loss['criterion']
                            }

                            tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='grid', workers=workers)

                            n_parameters = tr.parameters_count()
                            n_train = tr.database_size()

                            if n_parameters <= 0.12*n_train:
                                try:
                                    time.sleep(1)
                                    tr.start_training(save_plots=save_plots)

                                    tr.close_plots()
                                    del(tr)
                                except:
                                    pbar.update()
                                    continue

                            else:
                                time.sleep(0.3)

                            pbar.update()
                    
                del(pbar)

            print('\nGrid search complete...\n')
            print('+'*50)

        if 'optimization' in perform:
            print('\n', '+'*50)
            print('Performing optimization...\n')
            for hidden_size in range(start_point, max_hidden_layers+1):

                Network = self.Networks[hidden_size]

                print('.'*50)
                print(Network)
                print('.'*50, '\n')

                # Search for better optimizer and criterion over network
                if network:
                    print('Running specific network\n')
                    better_network = network
                else:
                    try:
                        rd = Reader(hidden_size, f'{hidden_size}Hlayer{extra_name}.csv', type='complete', step='grid')
                        better_network = rd.recover_best(n_networks=n_networks, criteria=reader_criteria)

                        if better_network == None:
                            print('Any functional model found for {}...')
                            break
                    
                    except:
                        print(f'No files found for {Network}')
                        continue
                
                for i, network_step in enumerate(better_network):
                    
                    if n_networks > 1:
                        print(f'Runing Network Test {i+1}/{n_networks}\n')

                    pbar = tqdm(total=len(optimizers)*len(loss_functions), desc='Optimizers', colour='green')

                    file = Network + f'{extra_name}.csv'

                    for optimizer in optimizers:

                        for criterion in loss_functions:

                            architecture = {
                                'model' : Network,
                                'num_targets' : num_targets,
                                'num_features' : num_features,
                                'dimension' : network_step['dimension'],
                                'activation_functions' : network_step['activation_functions'],
                                'optimizer' : optimizer,
                                'criterion' : criterion
                            }

                            tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='optimization', workers=workers)

                            n_parameters = tr.parameters_count()
                            n_train = tr.database_size()

                            if n_parameters <= 0.12*n_train:
                                time.sleep(1)

                                try:
                                    tr.start_training(save_plots=save_plots)

                                    tr.close_plots()
                                    del(tr)
                                
                                except:
                                    pbar.update()
                                    continue

                            pbar.update()
                    
                    del(pbar)
            
            print('\nOptimization search complete...\n')
            print('+'*50)

        if 'tuning_batch' in perform:
            
            print('\n', '+'*50)
            print('Performing tuning batch search...\n')
            
            for hidden_size in range(start_point, max_hidden_layers+1):

                Network = self.Networks[hidden_size]

                print('.'*50)
                print(f'{Network}')
                print('.'*50, '\n')
                    
                # Search for better batch size in network
                if network:
                    print('Running specific network\n')
                    better_network = network
                else:
                    try:
                        rd = Reader(hidden_size, f'{extra_name}.csv', type='complete', step='optimization')
                        better_network = rd.recover_best(n_networks=n_networks, criteria=reader_criteria)
                    except:
                        print(f'No files found for {Network}')
                        continue

                    if better_network == None:
                        print('Any functional model found for {}...')
                        break

                file = Network + f'{extra_name}_batches'
                rnd = np.random.RandomState(seed=seed)

                batches = rnd.randint(bz_range[0], bz_range[1], tries)

                file += '.csv'

                n_iterations = len(batches)
                
                for i, network_step in enumerate(better_network):

                    if n_networks > 1:
                        print(f'Runing Network Test {i+1}/{n_networks}\n')

                    pbar = tqdm(total=n_iterations, desc='Batches', colour='green')

                    for batch_size in batches:

                        architecture = {
                            'model' : Network,
                            'num_targets' : num_targets,
                            'num_features' : num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : network_step['activation_functions'],
                            'optimizer' : network_step['optimizer'],
                            'criterion' : network_step['criterion']
                        }

                        config.update(batch_size=int(batch_size))

                        tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='tuning_batch', workers=workers)

                        n_parameters = tr.parameters_count()
                        n_train = tr.database_size()

                        if n_parameters <= 0.12*n_train:
                            time.sleep(1)
                            try:
                                tr.start_training(save_plots=save_plots)

                                tr.close_plots()
                                del(tr)
                            except:
                                pbar.update()
                                continue
                            
                        pbar.update()
                    
                    del(pbar)

                ######################## Testing nearest powers of two in batches ##########################
                print('\n', '#'*16, ' Batch powering... ', '#'*16, '\n')
                
                try:
                    rd = Reader(hidden_size, f'{extra_name}_batches', type='complete', step='tuning_batch')
                    better_network = rd.recover_best(criteria=reader_criteria)
                except:
                    print(f'No files found for {Network}')
                    continue

                if better_network == None:
                    print('Any functional model found for {}...')
                    break
                else:
                    better_network = better_network[0]
                
                new_batches = self.get_batches(better_network['batch_size'])

                for batch_size in new_batches:

                    print(f'Batch size => {batch_size}\n')

                    architecture = {
                        'model' : Network,
                        'num_targets' : num_targets,
                        'num_features' : num_features,
                        'dimension' : better_network['dimension'],
                        'activation_functions' : better_network['activation_functions'],
                        'optimizer' : better_network['optimizer'],
                        'criterion' : better_network['criterion']
                    }

                    config.update(batch_size=int(batch_size))

                    tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='tuning_batch', workers=workers)

                    n_parameters = tr.parameters_count()
                    n_train = tr.database_size()

                    if n_parameters <= 0.12*n_train:
                        try:
                            time.sleep(1)
                            tr.start_training(save_plots=save_plots)

                            tr.close_plots()
                            del(tr)
                        except:
                            continue

            print('\nBatch search complete...\n')
            print('+'*50)

        if 'tuning_lr' in perform:
            
            print('\n', '+'*50)
            print('Performing learning rate search...\n')
            for hidden_size in range(start_point, max_hidden_layers+1):

                Network = self.Networks[hidden_size]

                print('.'*50)
                print(f'{Network}')
                print('.'*50, '\n')

                # Search for better learning_rate in network
                if network:
                    print('Running specific network\n')
                    better_network = network
                else:
                    try:
                        rd = Reader(hidden_size, f'{extra_name}_batches', type='complete', step='tuning_batch')
                        better_network = rd.recover_best(criteria=reader_criteria)
                    except:
                        print(f'No files found for {Network}')
                        continue

                    if better_network == None:
                        print('Any functional model found for {}...')
                        break

                file = Network + f'{extra_name}_lr'
                rnd = np.random.RandomState(seed=seed)

                learning_rates = rnd.uniform(lr_range[0], lr_range[1], tries)
                
                file += '.csv'

                n_iterations = len(learning_rates)

                for i, network_step in enumerate(better_network):
                    
                    if n_networks > 1:
                        print(f'Runing Network Test {i+1}/{n_networks}\n')

                    pbar = tqdm(total=n_iterations, desc='learning rates', colour='green')

                    for learning_rate in learning_rates:

                        architecture = {
                            'model' : Network,
                            'num_targets' : num_targets,
                            'num_features' : num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : network_step['activation_functions'],
                            'optimizer' : network_step['optimizer'],
                            'criterion' : network_step['criterion']
                        }

                        config.update(
                            batch_size=int(network_step['batch_size']), 
                            learning_rate=round(learning_rate, 9)
                            )

                        tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='tuning_lr', workers=workers)

                        n_parameters = tr.parameters_count()
                        n_train = tr.database_size()

                        if n_parameters <= 0.12*n_train:
                            try:
                                time.sleep(1)
                                tr.start_training(save_plots=save_plots)

                                tr.close_plots()
                                del(tr)
                            except:
                                pbar.update()
                                continue

                        pbar.update()
                    
                    del(pbar)
            
            print('\nLearning rate search complete...\n')
            print('+'*50)

        if 'lineal' in perform:
            print('\n', '+'*50)
            print('Performing linear searching...\n')
            for hidden_size in range(start_point, max_hidden_layers+1):

                Network = self.Networks[hidden_size]

                print('.'*50)
                print(Network)
                print('.'*50, '\n')

                # Search for better learning_rate in network
                if network:
                    print('Running specific network\n')
                    better_network = network
                else:
                    try:
                        rd = Reader(hidden_size, f'{extra_name}_lr', type='complete', step='tuning_lr')
                        better_network = rd.recover_best(criteria=reader_criteria)
                    except:
                        print(f'No files found for {Network}')
                        continue

                    if better_network == None:
                        print('Any functional model found for {}...')
                        break

                file = Network + f'{extra_name}_lr_lineal' #!!!!!!!!!!!!!!!!!!!
                file += '.csv'

                for i, network_step in enumerate(better_network):
                    
                    if n_networks > 1:
                        print(f'Runing Network Test {i+1}/{n_networks}\n')

                    def model_function(lr):
                        architecture = {
                            'model' : Network,
                            'num_targets' : num_targets,
                            'num_features' : num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : network_step['activation_functions'],
                            'optimizer' : network_step['optimizer'],
                            'criterion' : network_step['criterion']
                        }

                        config.update(
                            batch_size=int(network_step['batch_size']),
                            learning_rate=round(lr[-1], 9)
                        )

                        tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='tuning_lr', workers=workers)

                        n_parameters = tr.parameters_count()
                        n_train = tr.database_size()

                        if n_parameters <= 0.12*n_train:
                            time.sleep(1)
                            tr.start_training(save_plots=save_plots)

                            tr.close_plots()

                        if 'acc' in reader_criteria:
                            err = abs(1 - tr.values_plot['general_acc_val'][-1])
                        else:
                            err = abs(1 - tr.values_plot['general_r2_val'][-1])

                        print(f'{Network}: {lr[-1]} -> {err:.4f}')

                        return err

                    def run():
                        lr_local = network_step['lr']

                        if 'acc' in reader_criteria:
                            err_0 = abs(1 - network_step['acc']/100)
                        else:
                            err_0 = abs(1 - network_step['r2'])
                        
                        print(f"{Network}: {lr_local:.6f} -> {err_0:.4f}")

                        result = minimize(model_function, lr_local, method='Nelder-Mead')
                        return result
                    
                    result_lr = run()
                    better_lr = result_lr.x[0]
                    
                    print(f'\n Better lr -> {better_lr}')

            print('\nLinear search complete...\n')
            print('+'*50)

        if 'random_state' in perform:
            print('\n', '+'*50)
            print('Performing random_state searching...\n')
            for hidden_size in range(start_point, max_hidden_layers+1):

                Network = self.Networks[hidden_size]

                print('.'*50)
                print(f'{Network}')
                print('.'*50, '\n')

                # Search for better learning_rate in network
                if network:
                    print('Running specific network\n')
                    better_network = network
                else:
                    try:
                        rd = Reader(hidden_size, f'{extra_name}_lr', type='complete', step='tuning_lr')
                        better_network = rd.recover_best(criteria=reader_criteria)
                    except:
                        print(f'No files found for {Network}')
                        continue

                    if better_network == None:
                        print('Any functional model found for {}...')
                        break
                
                rnd = np.random.RandomState(seed=seed)
                random_states = rnd.randint(150, 200000, tries)

                file = Network + f'{extra_name}_lr_RS' #!!!!!!!!!!!!!!!!!!!

                file += '.csv'

                n_iterations = len(random_states)
                
                for i, network_step in enumerate(better_network):
                    
                    if n_networks > 1:
                        print(f'Runing Network Test {i+1}/{n_networks}\n')
                    
                    pbar = tqdm(total=n_iterations, desc='Random States', colour='green')

                    for RS in random_states:

                        architecture = {
                            'model' : Network,
                            'num_targets' : num_targets,
                            'num_features' : num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : network_step['activation_functions'],
                            'optimizer' : network_step['optimizer'],
                            'criterion' : network_step['criterion']
                        }

                        config.update(
                            batch_size = int(network_step['batch_size']),
                            learning_rate = round(float(network_step['lr']), 9),
                            random_state = RS
                        )

                        tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='random_state', workers=workers)

                        n_parameters = tr.parameters_count()
                        n_train = tr.database_size()

                        if n_parameters <= 0.12*n_train:
                            try:
                                time.sleep(1)
                                tr.start_training(save_plots=save_plots)

                                tr.close_plots()
                            except:
                                pbar.update()
                                continue

                        pbar.update()
                    
                    del(pbar)

            print('\nrandom_state search complete...\n')
            print('+'*50)
        
        if 'around_exploration' in perform:
            print('\n', '+'*50)
            print('Performing around exploration ...\n')
            for hidden_size in range(start_point, max_hidden_layers+1):

                Network = self.Networks[hidden_size]

                print('.'*50)
                print(f'{Network}')
                print('.'*50, '\n')

                # Search for better learning_rate in network
                if network:
                    print('Running specific network\n')
                    better_network = network
                else:
                    try:
                        rd = Reader(hidden_size, f'{extra_name}_lr', type='complete', step='random_state')
                        better_network = rd.recover_best(criteria=reader_criteria)
                    except:
                        print(f'No files found for {Network}')
                        continue

                    if better_network == None:
                        print('Any functional model found for {}...')
                        break

                file = Network + f'{extra_name}_RE' #!!!!!!!!!!!!!!!!!!!

                for i, network_step in enumerate(better_network):
                    
                    if n_networks > 1:
                        print(f'Runing Network Test {i+1}/{n_networks}\n')

                    initial_lr = float(network_step['lr'])
                    lr_list = []

                    initial_batch = int(network_step['batch_size'])
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

                    file += '.csv'
                    
                    n_values = len(lr_list)*len(batch_list)
                    pbar = tqdm(total=n_values, desc='Exploration steps', colour='green')

                    for lr in lr_list:
                        architecture = {
                            'model' : Network,
                            'num_targets' : num_targets,
                            'num_features' : num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : network_step['activation_functions'],
                            'optimizer' : network_step['optimizer'],
                            'criterion' : network_step['criterion']
                        }

                        for batch in batch_list:
                            config.update(
                                batch_size = batch,
                                learning_rate = round(lr, 9),
                                random_state = rs
                            )

                            tr = Trainer(file, architecture, config.hyperparameters, config, mode=mode, step='around_exploration', workers=workers)

                            n_parameters = tr.parameters_count()
                            n_train = tr.database_size()

                            if n_parameters <= 0.12*n_train:
                                try:
                                    time.sleep(1)
                                    tr.start_training(save_plots=save_plots)

                                    tr.close_plots()
                                except:
                                    pbar.update()
                                    continue

                            pbar.update()
                        
                    del(pbar)

                print('\naround_exploration search complete...\n')
                print('+'*50)