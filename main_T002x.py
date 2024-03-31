import nnTrainer
from nnTrainer.launch.LaunchBuilder import LaunchBuilder

if __name__=='__main__':
    # Define the grid searchin steps to compute
    perform = [
        nnTrainer.Grid,
        nnTrainer.Optimization,
        nnTrainer.RestartGridFromWorst,
        nnTrainer.Optimization,
        nnTrainer.RandomState,
        nnTrainer.TuningLr,
        nnTrainer.TuningBatch,
        nnTrainer.RandomState,
        nnTrainer.AroundExploration
    ]

    ##############
    # Parameters #
    ##############
    ID = 'T002x'

    # Configurator
    config = nnTrainer.Configurator()
    
    # Update config object with the required parameters
    config.update(
        max_hidden_layers = 6,
        min_neurons = 1,
        max_neurons = 10,
        n_tries = 150,
        n_networks = 1,
        start_point = 1,
        num_epochs = 1400,
        random_state=5502,
        batch_size=256,
        scale_y = True,
        learning_rate=0.001,
        lineal_output = False,
        reader_criteria = 'r2_val_general',
        seed = 1202,
        train_ID = ID,
        limit_threads = True,
        save_full_predictions = False,
        config_file = 'configT001x.json',
    )

    # Training launcher
    mb = LaunchBuilder(perform)

    # Launch the training
    mb.launch_training()