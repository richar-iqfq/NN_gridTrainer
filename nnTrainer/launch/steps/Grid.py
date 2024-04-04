import logging
import itertools
import copy

from tqdm import tqdm

from nnTrainer.launch.Main import MainLauncher
from nnTrainer.train.Trainer import Trainer
from nnTrainer.tools.Train import get_random_activation_functions

class Grid(MainLauncher):
    '''
    Grid step launcher
    '''
    def __init__(self):
        super().__init__()

        self.actual_step = 'Grid'

    def assembly(self) -> None:
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

            Network = self.build_network_name(hidden_size)
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
                            'num_layers' : hidden_size,
                            'num_targets' : self.num_targets,
                            'num_features' : self.num_features,
                            'dimension' : dimension,
                            'activation_functions' : af,
                            'optimizer' : self.config.get_loss('optimizer'),
                            'criterion' : self.config.get_loss('criterion')
                        }

                        tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                        train_flag, tr = self.launch(tr)

                        if not train_flag:
                            pbar.update()
                            continue

                        pbar.update()

                else:
                    better_network = self.recover_network(last_hidden_size, step=self.actual_step)

                    if better_network == None:
                        logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
                        print(f'Any functional model found for {hidden_size} hidden layers...')
                        continue
                    
                    for network_step in better_network:
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
                                'num_layers' : hidden_size,
                                'num_targets' : self.num_targets,
                                'num_features' : self.num_features,
                                'dimension' : dimension,
                                'activation_functions' : final_af,
                                'optimizer' : self.config.get_loss('optimizer'),
                                'criterion' : self.config.get_loss('criterion')
                            }

                            tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                            train_flag, tr = self.launch(tr)

                            if not train_flag:
                                pbar.update()
                                continue

                            pbar.update()

    def random(self) -> None:
        logging.info('Started ramdom mode')

        for hidden_size in range(self.start_point, self.max_hidden_layers+1):

            if hidden_size > 1:
                logging.info(f'Searching {hidden_size} layers')
                Network = self.build_network_name(hidden_size)
                file = Network + f'{self.extra_name}.csv'

                print('.'*50)
                print(Network)
                print('.'*50, '\n')

                better_network = self.recover_network(hidden_size, step=self.actual_step)

                if better_network == None:
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue
                
                for network_step in better_network:
                    final_af_list = get_random_activation_functions(self.af_list, hidden_size+1, self.lineal_output)
                    
                    total_architectures = len(final_af_list)

                    pbar = tqdm(total=total_architectures, desc='Architectures', colour='green')

                    for final_af in final_af_list:
                        architecture = {
                            'num_layers' : hidden_size,
                            'num_targets' : self.num_targets,
                            'num_features' : self.num_features,
                            'dimension' : network_step['dimension'],
                            'activation_functions' : final_af,
                            'optimizer' : self.config.get_loss('optimizer'),
                            'criterion' : self.config.get_loss('criterion')
                        }

                        tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                        train_flag, tr = self.launch(tr)

                        if not train_flag:
                            pbar.update()
                            continue
                        
                        pbar.update()

        logging.info('Grid search complete')
        print('\nGrid search complete...\n')
        print('+'*50)

    def run(self, previous_step: str, network: dict=False, modes: list=['assembly', 'random']) -> None:
        logging.info('Started grid search')

        print('+'*50)
        print('Performing grid search...\n')

        #====== Execute steps =====
        if 'assembly' in modes:
            self.assembly()

        if 'random' in modes:
            self.random()