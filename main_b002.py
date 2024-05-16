import nnTrainer
from nnTrainer.launch.LaunchBuilder import LaunchBuilder

if __name__=='__main__':
    # Define the grid searchin steps to compute
    perform = [
        # nnTrainer.Grid,
        # nnTrainer.Optimization,
        # nnTrainer.RestartGridFromWorst,
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
    # b = 1
    ID = 'b002'

    # Configurator
    config = nnTrainer.Configurator()
    
    # Update config object with the required parameters
    config.update(
        database = 'results_a-0.22.csv',
        max_hidden_layers = 20,
        min_neurons = 1,
        max_neurons = 6,
        n_tries = 150,
        n_networks = 1,
        start_point = 1,
        num_epochs = 800,
        random_state=1234,
        batch_size=256,
        learning_rate=0.001,
        lineal_output = True,
        seed = 22164,
        train_ID = ID,
        limit_threads = True,
        save_full_predictions = False,
        config_file = 'configb002.json'
    )

    # Training launcher
    mb = LaunchBuilder(perform)

    # Launch the training
    mb.launch_training()