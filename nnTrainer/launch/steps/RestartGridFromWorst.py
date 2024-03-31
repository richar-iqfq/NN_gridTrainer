import logging

from nnTrainer.launch.Main import MainLauncher
from nnTrainer.launch.steps.Grid import Grid

class RestartGridFromWorst(MainLauncher):
    '''
    Restart from worst step launcher
    '''
    def __init__(self):
        super().__init__()

        self.actual_step = 'RestartGridFromWorst'

        self.grid = Grid()

    def run(self, previous_step: str, network: dict=False) -> None:
        logging.info('Restarted from worst optimizer')
        print('\n', '+'*50)
        print('Performing grid from worst...\n')
    
        worst_network = self.recover_network(4, previous_step, worst=True)

        if worst_network == None:
            logging.info(f"Any functional model found for {4} hidden layers in {self.actual_step}")
            print(f'Any functional model found for {4} hidden layers...')

        self.config.update(
            optimizer = worst_network[0]['optimizer'],
            criterion = worst_network[0]['criterion']
        )

        # Run grid
        self.grid.run('', modes=['assembly', ''])