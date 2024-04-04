import logging

from tqdm import tqdm
import numpy as np

from nnTrainer.launch.Main import MainLauncher
from nnTrainer.train.Trainer import Trainer

class RandomState(MainLauncher):
    '''
    Random state launcher
    '''
    def __init__(self):
        super().__init__()

        self.actual_step = 'RandomState'

    def run(self, previous_step: str, network: dict=False) -> None:
        logging.info('Random state search started')

        print('\n', '+'*50)
        print('Performing random_state searching...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers')

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
                        'num_layers' : hidden_size,
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

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                    train_flag, tr = self.launch(tr)
                    
                    if not train_flag:
                        pbar.update()
                        continue

                    pbar.update()

        logging.info('Random state search complete')
        print('\nrandom_state search complete...\n')
        print('+'*50)