import logging

from scipy.optimize import minimize

from nnTrainer.launch.Main import MainLauncher
from nnTrainer.train.Trainer import Trainer

class Lineal(MainLauncher):
    def __init__(self):
        super().__init__()

        self.actual_step = 'Lineal'

    def run(self, previous_step: str, network: dict=False):
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
                    logging.info(f"Any functional model found for {hidden_size} hidden layers in {self.actual_step}")
                    print(f'Any functional model found for {hidden_size} hidden layers...')
                    continue

            file = Network + f'{self.extra_name}' #!!!!!!!!!!!!!!!!!!!
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

                    tr = Trainer(file, architecture, self.config.get_hyperparameters(), step=self.actual_step, workers=self.workers)

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