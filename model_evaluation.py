from modules.Valuator import Valuator
from modules.Configurator import Configurator

if __name__=='__main__':
    #====================== Inputs ======================================
    
    train_id = 'T000'
    step = 'tuning_batch'
    hidden_layers = 4
    
    criteria = 'outliers_general'
    #====================================================================
    
    config = Configurator()
    config.update(
        scale_y = True,
        reader_criteria = 'outliers_count',
        extra_filename = train_id,
        train_ID = train_id,
        save_full_predictions = True
    )

    valuator = Valuator(config, hidden_layers, step, criteria)
    valuator.run()