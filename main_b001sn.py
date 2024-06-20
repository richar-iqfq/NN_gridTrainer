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
        nnTrainer.RandomState,
        nnTrainer.AroundExploration
    ]

    ##############
    # Parameters #
    ##############
    # b = 3
    ID = 'b001sn'

    # Configurator
    config = nnTrainer.Configurator()
    
    # Update config object with the required parameters
    config.update(
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
        seed = 54891,
        train_ID = ID,
        limit_threads = True,
        save_full_predictions = False,
        config_file = 'configb001.json'
    )

    # Training launcher
    mb = LaunchBuilder(perform)

    network = [
        {
            'hidden_layers' : 5,
            'dimension' : (18, 18, 16, 18, 14),
            'activation_functions' : (
                'nn.SELU()', 
                'nn.ReLU()', 
                'nn.LeakyReLU()', 
                'nn.ELU()', 
                'nn.SELU()', 
                'None'
            ),
            'optimizer' : 'Adam',
            'criterion' : 'nn.L1Loss()',
            'random_state' : 83618,
            'num_epochs' : 800,
            'batch_size' : 46,
            'lr' : 0.001,
        },
    ]

    # Launch the training
    mb.launch_training(last_step=nnTrainer.RandomState)