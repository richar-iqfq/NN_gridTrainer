import os
import subprocess

from nnTrainer.data.Sql import SqlReader
import nnTrainer
import matplotlib.pyplot as plt

code = 'T004'
step = nnTrainer.TuningBatch
max_size = {
    'b001' :20,
    'b002' :20,
    'b003' :20,
    'b004' :20,
    'b005' :20,
    'b001sn' :7,
    'b002sn' :7,
    'b003sn' :7,
    'b004sn' :7,
    'b005sn' :7,
    'b006sn' :7,
    'b007sn' :15,
    'b008sn' : 10,
    'b009sn' : 10,
    'T001' :8,
    'T002' :4,
    'T002b' : 4,
    'T004' : 4,
}

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
    'b006sn' : 'results_a-0.272.csv',
    'b007sn' : 'results_a-0.272.csv',
    'b008sn' : 'results_a-0.22.csv',
    'b009sn' : 'results_a-0.33.csv',
    'T001' : 'dataset_final_sorted_3.1.0.csv',
    'T002' : 'dataset_final_sorted_3.1.0.csv',
    'T002b' : 'dataset_final_sorted_3.1.0.csv',
    'T004' : 'dataset_final_sorted_4.0.0.csv'
}

config = nnTrainer.Configurator()

config.update(
    database = database[code]
)

reader = SqlReader()

r2test_data = []
r2val_data = []
out_data = []
hidden_data = []

max_r2 = 0
max_hidden = 0
is_data = False

for hidden in range(1, max_size[code]):
    try:
        values = reader.recover_best(code, hidden, step, criteria='R2Val_i', n_values=1)
    except:
        values = None
    
    if values:
        is_data = True
        best = values[0]

        r2_test = best['r2_test']
        r2_val = best['r2_val']
        outliers = best['outliers']

        r2test_data.append(r2_test)
        r2val_data.append(r2_val)
        out_data.append(outliers)
        hidden_data.append(hidden)

        if r2_test > max_r2:
            max_r2 = r2_test
            max_hidden = hidden

if is_data:
    fig, ax = plt.subplots(1)

    fig.suptitle(f'{code} - {step}')

    ax.plot(hidden_data, r2test_data, marker='.')
    ax.plot(hidden_data, r2val_data, marker='+')
    ax.legend(['r2_test', 'r2_val'])
    ax.set_title('Correlation coefficient')
    ax.set_xlabel('Hidden size')
    ax.set_ylabel('Correlation coefficient')

    if len(r2test_data)>1:
        plt.show()

    print(f'Best: {max_hidden} layers -> r2_test: {max_r2}')

    values = reader.recover_best(code, max_hidden, step, criteria='R2Test_i', n_values=1)

    for network in values:
        for key in network:
            print(f' {key}: {network[key]}')

    pdf_path = os.path.join(values[0]['Path'], 'full_regression')
    pdf_unscalled_path = os.path.join(values[0]['Path'], 'full_unscalled_regression')

    if os.path.isdir(pdf_path):
        pdf_files = os.listdir(pdf_path)
    else:
        pdf_files = []
    
    if os.path.isdir(pdf_unscalled_path):
        pdf_unscalled_files = os.listdir(pdf_unscalled_path)
    else:
        pdf_unscalled_files = []

    if len(pdf_files) > 0:
        files = pdf_files
        path = pdf_path
    else:
        files = pdf_unscalled_files
        path = pdf_unscalled_path

    for file in files:
        pdf = os.path.join(path, file)
        subprocess.call(["xdg-open", pdf])
else:
    print('There are no data')