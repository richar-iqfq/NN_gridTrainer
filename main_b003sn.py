import nnTrainer
from nnTrainer.launch.LaunchBuilder import LaunchBuilder

if __name__=='__main__':
    # Define the grid searchin steps to compute
    perform = [
        # nnTrainer.Grid,
        # nnTrainer.Optimization,
        # nnTrainer.RestartGridFromWorst,
        # nnTrainer.Optimization,
        # nnTrainer.RandomState,
        nnTrainer.TuningLr,
        nnTrainer.TuningBatch,
        # nnTrainer.RandomState,
        nnTrainer.AroundExploration
    ]

    ##############
    # Parameters #
    ##############
    # b = 4
    ID = 'b003sn'

    # Configurator
    config = nnTrainer.Configurator()
    
    # Update config object with the required parameters
    config.update(
        max_hidden_layers = 6,
        min_neurons = 1,
        max_neurons = 6,
        n_tries = 300,
        n_networks = 1,
        start_point = 6,
        num_epochs = 800,
        random_state=1234,
        batch_size=256,
        learning_rate=0.001,
        lineal_output = True,
        seed = 88789,
        train_ID = ID,
        limit_threads = True,
        save_full_predictions = False,
        config_file = 'configb003.json'
    )

    # Training launcher
    mb = LaunchBuilder(perform)

    network = [
        {
            'hidden_layers' : 6,
            'dimension' : (12, 20, 11, 17, 17, 18),
            'activation_functions' : (
                'nn.ReLU()',
                'nn.ReLU()',
                'nn.ELU()',
                'nn.SELU()',
                'nn.ELU()',
                'nn.LeakyReLU()',
                'None'
            ),
            'optimizer' : 'Adam',
            'criterion' : 'nn.SmoothL1Loss()',
            'random_state' : 151699,
            'num_epochs' : 800,
            'batch_size' : 35,
            'lr' : 0.001,
        },
    ]

    # Launch the training
    mb.launch_training(last_step=nnTrainer.RandomState)