import logging

from tqdm import tqdm
import numpy as np

from nnTrainer.launch.Main import MainLauncher
from nnTrainer.train.Trainer import Trainer
from nnTrainer.tools.Train import get_adjacent_batches

class TuningBatch(MainLauncher):
    def __init__(self):
        super().__init__()

        self.actual_step = 'TuningBatch'

    def nearest_powers_of_two(self, hidden_size: int, file: str):
        logging.info(f'Searching nearest powers of two in batches (Parted: {self.parted})')
        print('\n', '#'*16, ' Batch powering... ', '#'*16, '\n')
        
        better_network = self.recover_network(hidden_size, step=self.actual_step)

        if better_network == None:
            logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
            print(f'Any functional model found for {hidden_size} hidden layers...')
            return False
        
        pbar = tqdm(total=2, desc='Batches', colour='green')

        for i, network_step in enumerate(better_network):
            for batch_size in get_adjacent_batches(network_step['batch_size']):

                print(f'Batch size => {batch_size}\n')

                architecture = {
                    'num_layers' : hidden_size,
                    'num_targets' : self.num_targets,
                    'num_features' : self.num_features,
                    'dimension' : network_step['dimension'],
                    'activation_functions' : network_step['activation_functions'],
                    'optimizer' : network_step['optimizer'],
                    'criterion' : network_step['criterion']
                }

                self.config.update(batch_size=int(batch_size))

                tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                train_flag, tr = self.launch(tr)

                if not train_flag:
                    pbar.update()
                    continue
    
    def three_increments_and_decrements(self, hidden_size: int, file: str):
        logging.info(f'Searching increments in batches (Parted: {self.parted})')
        print('\n', '#'*16, ' Testing three increments... ', '#'*16, '\n')
        
        better_network = self.recover_network(hidden_size, step=self.actual_step)

        if better_network == None:
            logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
            print(f'Any functional model found for {hidden_size} hidden layers...')
            return False

        pbar = tqdm(total=6, desc='Batches', colour='green')

        for i, network_step in enumerate(better_network):
            new_batches = []

            for j in range(1, 4):
                new_batches.append(network_step['batch_size']+j)
                new_batches.append(network_step['batch_size']-j)

            for batch_size in new_batches:

                print(f'Batch size => {batch_size}\n')

                architecture = {
                    'num_layers' : hidden_size,
                    'num_targets' : self.num_targets,
                    'num_features' : self.num_features,
                    'dimension' : network_step['dimension'],
                    'activation_functions' : network_step['activation_functions'],
                    'optimizer' : network_step['optimizer'],
                    'criterion' : network_step['criterion']
                }

                self.config.update(batch_size=int(batch_size))

                tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                train_flag, tr = self.launch(tr)

                if not train_flag:
                    pbar.update()
                    continue

    def run(self, previous_step: str, network: dict=False, extra: list=['nearest_powers_of_two', 'three_increments_and_decrements']) -> None:
        logging.info(f'Tuning batch search started (Parted: {self.parted})')
        print('\n', '+'*50)
        print('Performing tuning batch search...\n')
        
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers (Parted: {self.parted})')
            Network = self.build_network_name(hidden_size)

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
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
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
                        'num_layers' : hidden_size,
                        'num_targets' : self.num_targets,
                        'num_features' : self.num_features,
                        'dimension' : network_step['dimension'],
                        'activation_functions' : network_step['activation_functions'],
                        'optimizer' : network_step['optimizer'],
                        'criterion' : network_step['criterion']
                    }

                    self.config.update(batch_size=int(batch_size))

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)

                    if not train_flag:
                        pbar.update()
                        continue
                        
                    pbar.update()

            ######################## Testing nearest powers of two in batches ##########################
            if 'nearest_powers_of_two' in extra:
                self.nearest_powers_of_two(hidden_size, file)

            ######################## Testing three increments and decrements in batches ##########################
            if 'three_increments_and_decrements' in extra:
                self.three_increments_and_decrements(hidden_size, file)
        
        logging.info(f'Tuning batch search complete (Parted: {self.parted})')
        print('\nBatch search complete...\n')
        print('+'*50)