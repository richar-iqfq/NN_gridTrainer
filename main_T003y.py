import nnTrainer

if __name__=='__main__':
    # Define the grid searchin steps to compute
    steps = [
        'grid',
        'optimization', 
        # 'restart_grid_from_worst',
        # 'optimization',
        # 'random_state',
        # 'tuning_lr',
        # 'tuning_batch', 
        # 'random_state',
        # 'around_exploration'
    ]

    ##############
    # Parameters #
    ##############
    ID = 'T003x'

    # Configurator
    config = nnTrainer.Configurator()
    
    # Update config object with the required parameters
    config.update(
        database = 'dataset_final_sorted_3.0.0.csv',
        max_hidden_layers = 6,
        min_neurons = 1,
        max_neurons = 10,
        n_tries = 150,
        n_networks = 1,
        start_point = 1,
        num_epochs = 1400,
        random_state=5582,
        batch_size=256,
        scale_y = True,
        learning_rate=0.001,
        lineal_output = False,
        reader_criteria = 'r2_val_general',
        seed = 85201,
        train_ID = ID,
        limit_threads = True,
        save_full_predictions = False,
        config_file = 'configT001x.json'
    )

    # Training launcher
    mb = nnTrainer.Launcher(steps)

    # Launch the training
    mb.Run()