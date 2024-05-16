import nnTrainer
from nnTrainer.config.Configurator import Configurator
from nnTrainer.train.Valuator import Valuator

if __name__=='__main__':
    #====================== Inputs ======================================
    
    code = 'b004sn'
    step = nnTrainer.Optimization
    hidden_layers = 5
    
    criteria = 'R2Test_i'
    #====================================================================
    
    database = {
        'b001' : 'results_a-0.2.csv',
        'b002' : 'results_a-0.22.csv',
        'b003' : 'results_a-0.261.csv',
        'b004' : 'results_a-0.27.csv',
        'b005' : 'results_a-0.33.csv',
        'b001sn' : 'results_a-0.2.csv',
        'b002sn' : 'results_a-0.22.csv',
        'b003sn' : 'results_a-0.261.csv',
        'b004sn' : 'results_a-0.27.csv',
        'b005sn' : 'results_a-0.33.csv',
        'T001' : 'dataset_final_sorted_3.1.0.csv',
        'T002' : 'dataset_final_sorted_3.1.0.csv',
        'T002b' : 'dataset_final_sorted_3.1.0.csv',
    }

    config_files = {
        'b001sn' : 'configb001.json',
        'b002sn' : 'configb002.json',
        'b003sn' : 'configb003.json',
        'b004sn' : 'configb004.json',
        'b005sn' : 'configb005.json',
        'T001' : 'dataset_final_sorted_3.1.0.csv',
        'T002' : 'dataset_final_sorted_3.1.0.csv',
        'T002b' : 'dataset_final_sorted_3.1.0.csv',
    }

    config = Configurator()
    config.update(
        database = database[code],
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