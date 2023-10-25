import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_correlation_matrix(database, columns_to_drop, output_folder, save=False, show=False):
        '''
        Plot correlation matrix

        save (`bool`):
            if True, saves the fig_A.
            Default is False
        
        show(`bool`):
            if True, shows the builded plot. Default is False
        '''

        database_df = pd.read_csv(database)
        df = database_df.drop(columns=columns_to_drop)

        corr_m = df.corr(numeric_only=True)

        fig, ax = plt.subplots(1)
        fig.suptitle('Correlation Matrix', fontweight ="bold")
        fig.set_size_inches(9, 9)

        plt.subplots_adjust(bottom=0.215)

        sns.heatmap(corr_m, linewidths=0.5, mask=(np.abs(corr_m) <= 0.3), annot=True, annot_kws={"size":6}, square=True, ax=ax)
        
        if show:
            plt.show()

        if save:
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            
            file = os.path.join(output_folder, 'Correlation_matrix.pdf')
            fig.savefig(file, dpi=450, format='pdf')

        plt.close()

if __name__=='__main__':
    database = 'dataset/dataset_final_sorted_2.4.3.csv'

    columns = [
        "ID", "FORMULA", "NAME", "atom", "atom1", "atom2", "atom3", "atom4", "atom5", "atom6", "atom7", "atom8", "atom9",
        "atom10", "atom11", "atom12", "atom13"
    ]

    output_folder = 'dataset/'

    plot_correlation_matrix(database, columns, output_folder, save=True, show=True)
