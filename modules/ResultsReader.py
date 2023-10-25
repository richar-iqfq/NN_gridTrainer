import pandas as pd
import os

class Reader():
    '''
    Better neural network aschitecture reader

    Parameters
    ----------
    hidden_layers (`int`):
        Number of hidden layers for model

    extra_keyword (`str`):
        Extra keywords on file

    type (`str`):
        Training type: 'complete'

    Methods
    -------
    recover_best(n_networks=1):
         Returns a dictionary with the better trained networks
    '''
    def __init__(self, hidden_layers, extra_keyword, type='complete', step='grid'):
        self.name = f'Net_{hidden_layers}Hlayer'
        self.type = type

        self.path_name = {
            'grid' : '01_grid',
            'optimization' : '02_optimization',
            'tuning_batch' : '03_tuning_batch',
            'tuning_lr' : '04_tuning_lr',
            'lineal' : '05_lineal',
            'random_state' : '06_random_state',
            'around_exploration' : '07_around_exploration',
            'recovering' : 'recovering' 
        }

        path = os.path.join('Training_results', type, self.path_name[step])
    
        files = [file for file in os.listdir(path) if extra_keyword in file]

        self.files = [os.path.join(path, file) for file in files if self.name in file]

        if len(self.files) == 0:
            raise Exception('Any files found for the given structure')

    def recover_best(self, n_networks=1, criteria='acc_val_general'):
        '''
        Search the better networks for the training type given

        Parameters
        ----------
        n_networks (`int`):
            Number of networks to return in dict

        Returns
        -------
            Dictionary with the better trained networks
        '''
        df_l = []

        # Load csv files
        for i, file in enumerate(self.files):
            df_l.append(pd.read_csv(file))

        # build a single dataframe
        total_df = df_l[0]
        for df in df_l[1::]:
            total_df = pd.concat([total_df, df], axis=0)

        # Filter by validation results
        total_df = total_df[(total_df['MAE_val_general'] <= 1) & (total_df['acc_val_general'] > 0) & (total_df['r2_val_general'] < 1)]

        # Sort results by criteria
        if criteria != 'outliers_count':
            total_df = total_df.sort_values([criteria], ignore_index=True, ascending=False)
        else:
            total_df = total_df.sort_values([criteria], ignore_index=True, ascending=True)

        index = [i for i in range(n_networks)]

        # Check if total_df is not empty
        if len(total_df) == 0:
            print('Any row pass the filters...')
            return None
        
        # Select row
        sel_df = total_df.iloc[index]

        # Build list to store data
        better_networks = [0]*n_networks

        for i in range(n_networks):
            dimension = sel_df['dimension'].iloc[i]
            dimension = dimension.replace('|', ',')

            architecture = sel_df['architecture'].iloc[i]
            architecture = architecture.replace('|', ', ')
            
            optimizer = sel_df['optimizer'].iloc[i]
            loss_function = sel_df['loss_function'].iloc[i]

            epochs = sel_df['epochs'].iloc[i]
            batch = sel_df['batch_size'].iloc[i]
            lr = sel_df['lr'].iloc[i]
            learning_rate = str(lr)[0:9]
            acc = sel_df['acc_val_general'].iloc[i]
            r2 = sel_df['r2_val_general'].iloc[i]
            outliers_count = sel_df['outliers_general'].iloc[i]
            random_state = sel_df['random_state'].iloc[i]

            better_networks[i] = {
                'hidden_layers' : len(eval(dimension)),
                'dimension' : eval(dimension),
                'activation_functions' : eval(architecture),
                'optimizer' : optimizer,
                'criterion' : loss_function,
                'random_state' : random_state,
                'num_epochs' :epochs,
                'batch_size' : batch,
                'lr' : float(learning_rate),
                'acc' : acc,
                'r2' : r2,
                'outliers' : outliers_count
            }
        
        return better_networks