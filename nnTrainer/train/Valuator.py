import os

from nnTrainer.config.Configurator import Configurator
from nnTrainer.data.Sql import SqlReader
from nnTrainer.train.Trainer import Trainer

class Valuator():
    def __init__(self, launch_code, hidden_layers, train_step, criteria) -> None:
        self.config = Configurator()

        self.extra_name: str = self.config.get_custom('extra_filename')

        self.launch_code = launch_code
        self.hidden_layers = hidden_layers
        self.train_step = train_step
        self.criteria = criteria

        self.num_features: int = self.config.get_json('num_features')
        self.num_targets: int = self.config.get_json('num_targets')

        self.reader = SqlReader()

        networks = self.reader.recover_best(
            self.launch_code,
            self.hidden_layers,
            self.train_step,
            self.criteria
        )

        self.network = networks[0]

        print('Network retrieved:')
        for key in self.network:
            print(f'{key}: {self.network[key]}')

        self.model_path = os.path.join(self.network['Path'], 'model.pth')

        self.architecture = {
            'num_layers' : self.hidden_layers,
            'num_targets' : self.num_targets,
            'num_features' : self.num_features,
            'dimension' : self.network['dimension'],
            'activation_functions' : self.network['activation_functions'],
            'optimizer' : self.network['optimizer'],
            'criterion' : self.network['criterion']
        }

        self.config.update(
            save_full_predictions = True,
            batch_size = self.network['batch_size'],
            learning_rate = self.network['lr'],
            random_state = self.network['random_state']
        )

        self.trainer = Trainer(
            self.extra_name,
            self.architecture,
            self.config.get_hyperparameters(),
            'Recovering'
        )

        self.trainer.load_model(self.model_path)

    def evaluate_model(self):
        # Evaluate full data with trainer
        x, y, y_pred = self.trainer.eval_full_data()

        # Create routes
        self.trainer.create_routes()

        # Save full predictions
        self.trainer.write_full_predictions(x, y_pred)

        outliers = self.trainer.plots_builder.build_full_plots(y, y_pred)

        print(f'Outliers: {outliers}')

        self.trainer.plots_builder.show_plots()