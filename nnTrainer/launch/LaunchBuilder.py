import os

import numpy as np

import nnTrainer.launch.steps as steps
from nnTrainer.data.Sql import SqlDatabase

class LaunchBuilder():
    def __init__(self, perform: list) -> None:
        self.perform = perform

        # Record launch to database
        self.database_recorder = SqlDatabase()
        self.database_recorder.create_launch_record(perform)

    def launch_training(self, network: dict=False, last_step: str=False):
        os.system('clear')
        np.seterr(all="ignore")

        for i, step in enumerate(self.perform):
            if step == 'Grid':
                previous_step = last_step
            
            else:
                if 'Worst' in step:
                    previous_step = 'Grid'

                elif step == 'Optimization':
                    if last_step:
                        previous_step = last_step
                    else:
                        previous_step = 'Grid'

                else:
                    if i-1 >= 0:
                        previous_step = self.perform[i-1]
                    else:
                        previous_step = last_step

            # Define step function
            stepFunction = eval(f'steps.{step}()')

            # Execute function
            stepFunction.run(previous_step, network)