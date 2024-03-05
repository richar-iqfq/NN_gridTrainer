from modules.Valuator import Valuator
from modules.Configurator import Configurator

if __name__=='__main__':
    #====================== Inputs ======================================
    
    train_id = 'T00X'
    step = 'explore_lr'
    hidden_layers = 3
    
    criteria = 'acc_val_general'
    #====================================================================
    
    config = Configurator()
    config.update(
        scale_y = True,
        reader_criteria = 'outliers_count',
        extra_filename = train_id,
        train_ID = train_id,
        save_full_predictions = False
    )

    valuator = Valuator(config, hidden_layers, step, criteria)
    valuator.run()