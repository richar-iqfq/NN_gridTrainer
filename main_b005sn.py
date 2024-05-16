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
        # nnTrainer.TuningLr,
        nnTrainer.TuningBatch,
        nnTrainer.RandomState,
        nnTrainer.AroundExploration
    ]

    ##############
    # Parameters #
    ##############
    # b = 1
    ID = 'b005sn'

    # Configurator
    config = nnTrainer.Configurator()
    
    # Update config object with the required parameters
    config.update(
        database = 'results_a-0.33.csv',
        max_hidden_layers = 5,
        min_neurons = 1,
        max_neurons = 6,
        n_tries = 300,
        n_networks = 1,
        start_point = 5,
        num_epochs = 800,
        random_state=1234,
        batch_size=256,
        learning_rate=0.001,
        lineal_output = True,
        seed = 821319,
        train_ID = ID,
        limit_threads = True,
        save_full_predictions = False,
        config_file = 'configb005.json'
    )

    # Training launcher
    mb = LaunchBuilder(perform)

    network = [
        {
            'hidden_layers' : 5,
            'dimension' : (19, 17, 20, 20, 6),
            'activation_functions' : (
                'nn.Tanhshrink()',
                'nn.ReLU()',
                'nn.LeakyReLU()',
                'nn.ELU()',
                'nn.Tanhshrink()',
                'None'
            ),
            'optimizer' : 'Adam',
            'criterion' : 'nn.SmoothL1Loss()',
            'random_state' : 141467,
            'num_epochs' : 800,
            'batch_size' : 128,
            'lr' : 0.001,
        },
    ]

    # Launch the training
    mb.launch_training(network=network, last_step=nnTrainer.Grid)