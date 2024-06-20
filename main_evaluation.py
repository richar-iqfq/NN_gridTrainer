import nnTrainer
from nnTrainer.config.Configurator import Configurator
from nnTrainer.train.Valuator import Valuator

if __name__=='__main__':
    #====================== Inputs ======================================
    
    code = 'b008sn'
    step = nnTrainer.TuningBatch
    hidden_layers = 5
    
    criteria = 'R2Val_i'
    #====================================================================

    config_files = {
        'b001sn' : 'configb001.json',
        'b002sn' : 'configb002.json',
        'b003sn' : 'configb003.json',
        'b004sn' : 'configb004.json',
        'b005sn' : 'configb005.json',
        'b006sn' : 'configb006.json',
        'b008sn' : 'configb008.json',
        'b009sn' : 'configb009.json',
        'T001' : 'dataset_final_sorted_3.1.0.csv',
        'T002' : 'dataset_final_sorted_3.1.0.csv',
        'T002b' : 'dataset_final_sorted_3.1.0.csv',
    }

    config = Configurator()
    config.update(
        config_file = config_files[code],
        lineal_output = True,
        reader_criteria = criteria,
        train_ID = code,
    )

    valuator = Valuator(
        code,
        hidden_layers,
        step,
        criteria
    )

    valuator.evaluate_model()