import os

import numpy as np

import nnTrainer.launch.steps as steps
from ..base_class.Module import Module
from nnTrainer.data.Sql import SqlDatabase
from .. import (
    Grid,
    Optimization,
    TuningLr,
    TuningBatch,
    RandomState,
    Lineal,
    AroundExploration
)

class LaunchBuilder():
    def __init__(self, perform: list, modules: list = None) -> None:
        self.perform = perform
        self.modules = modules

        if modules:
            # Register new modules
            self.modules = modules

        # Record launch to database
        self.database_recorder = SqlDatabase()
        self.database_recorder.create_launch_record(perform)

    def launch_training(self, network: dict=False, last_step: str=False):
        os.system('clear')
        np.seterr(all="ignore")

        step_dict = {
            Grid : steps.Grid,
            Optimization : steps.Optimization,
            TuningLr : steps.TuningLr,
            TuningBatch : steps.TuningBatch,
            RandomState : steps.RandomState,
            Lineal : steps.Lineal,
            AroundExploration : steps.AroundExploration
        }

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
            stepFunction = step_dict[step]()

            # Execute before action
            if self.modules:
                for module in self.modules:
                    module.before(step)

            # Execute function
            stepFunction.run(previous_step, network)

            # Execute after action
            if self.modules:
                for module in self.modules:
                    module.after(step)