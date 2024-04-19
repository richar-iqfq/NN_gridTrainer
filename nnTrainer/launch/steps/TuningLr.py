import logging

from tqdm import tqdm
import numpy as np

from nnTrainer.base_class.Launcher import MainLauncher
from nnTrainer.train.Trainer import Trainer

class TuningLr(MainLauncher):
    def __init__(self):
        super().__init__()

        self.actual_step = 'TuningLr'

    def run(self, previous_step: str, network: dict=False):
        logging.info(f'Learning rate search started (Parted: {self.parted})')

        print('\n', '+'*50)
        print('Performing learning rate search...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers (Parted: {self.parted})')

            Network = self.build_network_name(hidden_size)

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
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
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
                        'num_layers' : hidden_size,
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
                    
                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)

                    if not train_flag:
                        pbar.update()
                        continue

                    pbar.update()
        
        logging.info(f'Learning rate search complete (Parted: {self.parted})')
        print('\nLearning rate search complete...\n')
        print('+'*50)