import logging

from tqdm import tqdm

from nnTrainer.base_class.Launcher import MainLauncher
from nnTrainer.train.Trainer import Trainer

class AddLayers(MainLauncher):
    '''
    Add layers to specific neural network
    '''
    def __init__(self):
        super().__init__()

        self.actual_step = 'AddLayers'
    
    def run(self, previous_step: str, network: dict=False) -> None:
        logging.info('Started AddLayers search')

        print('\n', '+'*50)
        print('Performing AddLayers...\n')

        print('Running specific network\n')
        better_network = network

        for i, network_step in enumerate(better_network):

            start_hidden_size = network_step['hidden_size']

            if self.n_networks > 1:
                print(f'Runing Network Test {i+1}/{self.n_networks}\n')

            for hidden_size in range(start_hidden_size, self.max_hidden_layers+1):
                logging.info(f'Searching {hidden_size} layers')
                Network = self.build_network_name(hidden_size)

                print('.'*50)
                print(Network)
                print('.'*50, '\n')

                pbar = tqdm(total=len(self.optimizers)*len(self.loss_functions), desc='Optimizers', colour='green')

                file = Network + f'{self.extra_name}.csv'

                ### Logic ###
            
        
