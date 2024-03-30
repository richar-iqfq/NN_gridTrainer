import os

import pandas as pd

class Reader():
    '''
    Better neural network aschitecture reader

    Parameters
    ----------
    hidden_layers (`int`):
        Number of hidden layers for model

    extra_keyword (`str`):
        Extra keywords on file

    Methods
    -------
    recover_best(n_values=1):
         Returns a dictionary with the better trained networks
    '''
    def __init__(self, hidden_layers, extra_keyword, step='grid'):
        self.name = f'Net_{hidden_layers}Hlayer'

        self.path_name = {
            'explore_lr' : '00_explore_lr',
            'grid' : '01_grid',
            'optimization' : '02_optimization',
            'tuning_batch' : '03_tuning_batch',
            'tuning_lr' : '04_tuning_lr',
            'lineal' : '05_lineal',
            'random_state' : '06_random_state',
            'around_exploration' : '07_around_exploration',
            'recovering' : 'recovering' 
        }

        path = os.path.join('Training_results', self.path_name[step])
    
        files = [file for file in os.listdir(path) if extra_keyword in file and '.~lock' not in file]

        self.files = [os.path.join(path, file) for file in files if self.name in file]

        if len(self.files) == 0:
            raise Exception('Any files found for the given structure')

    def recover_best(self, n_values=1, criteria='acc_val_general', worst=False):
        '''
        Search the better networks for the training type given

        Parameters
        ----------
        n_values (`int`):
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
        for column in total_df.columns:
            if 'r2' in column:
                total_df = total_df[(total_df[column] < 1)]
            if 'acc' in column:
                total_df = total_df[(total_df[column] > 0)]

        # Sort results by criteria
        if criteria != 'outliers_count':
            total_df = total_df.sort_values([criteria], ignore_index=True, ascending=False)
        else:
            total_df = total_df.sort_values([criteria], ignore_index=True, ascending=True)

        index = [i for i in range(n_values)]

        # Check if total_df is not empty
        if len(total_df) == 0:
            print('Any row pass the filters...')
            return None
        
        # Select row
        sel_df = total_df.iloc[index]

        # Build list to store data
        better_networks = [0]*n_values

        for i in range(n_values):
            if worst:
                j = -i
            else:
                j = i
            
            dimension = sel_df['dimension'].iloc[j]
            dimension = dimension.replace('|', ',')

            architecture = sel_df['architecture'].iloc[j]
            architecture = architecture.replace('|', ', ')
            
            optimizer = sel_df['optimizer'].iloc[j]
            loss_function = sel_df['loss_function'].iloc[j]

            epochs = sel_df['epochs'].iloc[j]
            batch = sel_df['batch_size'].iloc[j]
            lr = sel_df['lr'].iloc[j]
            learning_rate = str(lr)[0:9]
            acc = sel_df['acc_val_general'].iloc[j]
            r2 = sel_df['r2_val_general'].iloc[j]
            outliers_count = sel_df['outliers_general'].iloc[j]
            random_state = sel_df['random_state'].iloc[j]

            better_networks[j] = {
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