from nnTrainer.TrainingLauncher import Launcher
from nnTrainer.Configurator import Configurator

if __name__=='__main__':
    # Define the grid searchin steps to compute
    steps = [
        'explore_lr'
    ]

    # Training launcher
    mb = Launcher()

    # Configurator
    config = Configurator()
    
    # Update config object with the required parameters
    config.update(
        max_hidden_layers = 6,
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
        train_ID = 'T00X',
        limit_threads = True,
        save_full_predictions = False,
    )

    # Launch the training
    mb.Run_training(config, perform=steps)