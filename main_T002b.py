import nnTrainer
from nnTrainer.launch.LaunchBuilder import LaunchBuilder
from nnTrainer.data.Sql import SqlReader as SqlReader

if __name__=='__main__':
    # Define the grid searchin steps to compute
    perform = [
        nnTrainer.TuningLr,
        nnTrainer.TuningBatch,
        nnTrainer.AroundExploration
    ]

    ##############
    # Parameters #
    ##############
    ID = 'T002b'

    # Configurator
    config = nnTrainer.Configurator()

    # Update config object with the required parameters
    config.update(
        database = 'dataset_final_sorted_3.1.0.csv',
        max_hidden_layers = 3,
        min_neurons = 1,
        max_neurons = 40,
        n_tries = 600,
        n_networks = 1,
        start_point = 3,
        num_epochs = 1200,
        random_state=87282,
        batch_size=256,
        learning_rate=0.001,
        lineal_output = False,
        seed = 8500,
        train_ID = ID,
        limit_threads = True,
        save_full_predictions = False,
        config_file = 'configT002x.json',
        monitoring_state = True
    )

    # Training launcher
    mb = LaunchBuilder(perform)
    reader = SqlReader()
    network = reader.recover_best('T002', 3, nnTrainer.RandomState, criteria='R2Test_i', n_values=1)

    print(network)

    # Launch the training
    mb.launch_training(network=network)