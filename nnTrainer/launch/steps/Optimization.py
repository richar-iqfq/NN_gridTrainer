import logging

from tqdm import tqdm

from nnTrainer.base_class.Launcher import MainLauncher
from nnTrainer.train.Trainer import Trainer

class Optimization(MainLauncher):
    '''
    Optimization step launcher
    '''
    def __init__(self):
        super().__init__()

        self.actual_step = 'Optimization'

    def run(self, previous_step: str, network: dict=False) -> None:
        logging.info('Started optimization search')

        print('\n', '+'*50)
        print('Performing optimization...\n')
        for hidden_size in range(self.start_point, self.max_hidden_layers+1):
            logging.info(f'Searching {hidden_size} layers')
            Network = self.build_network_name(hidden_size)

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
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
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
                            'num_layers' : hidden_size,
                            'num_targets' : self.num_targets,
                            'num_features' : self.num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : network_step['activation_functions'],
                            'optimizer' : optimizer,
                            'criterion' : criterion
                        }

                        tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                        train_flag, tr = self.launch(tr)

                        if not train_flag:
                            pbar.update()
                            continue

                        pbar.update()
        
        logging.info('Optimization search complete')
        print('\nOptimization search complete...\n')
        print('+'*50)