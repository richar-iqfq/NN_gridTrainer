from modules.TrainingLauncher import Launcher
from modules.Configurator import Configurator

if __name__=='__main__':
    # Define the grid searchin steps to compute
    steps = [
        'grid', 'optimization', 'tuning_batch', 'tuning_lr', 
        'lineal', 'random_state', 'around_exploration'
    ]

    # Training launcher
    mb = Launcher()

    # Configurator
    config = Configurator()
    
    # Update config object with the required parameters
    config.update(
        max_hidden_layers = 6,
        min_neurons = 1,
        max_neurons = 10,
        n_tries = 10,
        n_networks = 1,
        start_point = 1,
        num_epochs = 1000,
        scale_y = True,
        lineal_output = False,
        reader_criteria = 'r2_val_general',
        seed = 85201,
        train_ID = 'T000',
        limit_threads = True,
        save_full_predictions = False,
    )

    # Launch the training
    mb.Run_training(config, perform=steps)