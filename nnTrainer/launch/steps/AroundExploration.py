import logging

from tqdm import tqdm

from nnTrainer.base_class.Launcher import MainLauncher
from nnTrainer.train.Trainer import Trainer

class AroundExploration(MainLauncher):
    def __init__(self):
        super().__init__()

        self.actual_step = 'AroundExploration'

    def run(self, previous_step: str, network: dict=False) -> None:
        logging.info('Around exploration started')

        print('\n', '+'*50)
        print('Performing around exploration ...\n')
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
                        'num_layers' : hidden_size,
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

                        tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

                        train_flag, tr = self.launch(tr)

                        if not train_flag:
                            pbar.update()
                            continue

                        pbar.update()

            logging.info('Around exploration complete')
            print('\naround_exploration search complete...\n')
            print('+'*50)