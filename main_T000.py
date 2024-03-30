import nnTrainer

if __name__=='__main__':
    # Define the grid searchin steps to compute
    steps = [
        'grid', 'optimization', 'tuning_batch', 'tuning_lr', 
        'lineal', 'random_state', 'around_exploration'
    ]

    # Configurator
    config = nnTrainer.Configurator()
    
    # Update config object with the required parameters
    config.update(
        max_hidden_layers = 6,
        min_neurons = 1,
        max_neurons = 10,
        n_tries = 10,
        n_networks = 1,
        start_point = 1,
        num_epochs = 1200,
        random_state=558,
        batch_size=256,
        scale_y = True,
        learning_rate=0.01,
        lineal_output = False,
        reader_criteria = 'r2_val_general',
        seed = 85201,
        train_ID = 'T000',
        limit_threads = True,
        save_full_predictions = False,
    )

    # Training launcher
    mb = nnTrainer.Launcher()

    # Launch the training
    mb.Run_training(config, perform=steps)