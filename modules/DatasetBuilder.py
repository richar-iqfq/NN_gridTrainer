import torch
from torch.utils.data import Dataset

from modules.PreprocessData import PreprocessData

#================================ Dataset preparation =================================
def create_datasets() -> tuple:
    '''
    Main function to build the Dataset classes for training.

    Returns
    -------
    train_dataset: Pytorch training dataset class

    val_dataset: Pytorch validation dataset class

    test_dataset: Pytorch test dataset class
    '''
    # Processer
    processer = PreprocessData()

    x_train, x_val, x_test, y_train, y_val, y_test = processer.Retrieve_Splitted()

    # Create the training dataset class
    class Mol_TrainDataset(Dataset):
        '''
        Create the Pytorch Train dataset
        '''

        def __init__(self):
            # Initialize data
            self.n_samples = len(y_train)

            # Select the x data
            self.x_data = torch.from_numpy(x_train) # size [n_samples, n_features]
            self.x_data = self.x_data.to(torch.float32)

            # Select the y data
            self.y_data = torch.from_numpy(y_train) # size [n_samples, 1]
            self.y_data = self.y_data.to(torch.float32)

        # Indexing
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        # len(dataset) to return the size
        def __len__(self):
            return self.n_samples

    # Create the validation dataset class
    class Mol_ValDataset(Dataset):
        '''
        Create the Pytorch Val dataset
        '''
        def __init__(self):
            # Initialize data
            self.n_samples = len(y_val)

            # Select the x data
            self.x_data = torch.from_numpy(x_val) # size [n_samples, n_features]
            self.x_data = self.x_data.to(torch.float32)

            # Select the y data
            self.y_data = torch.from_numpy(y_val) # size [n_samples, 1]
            self.y_data = self.y_data.to(torch.float32)

        # Indexing
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        # len(dataset) to return the size
        def __len__(self):
            return self.n_samples

    # Create the test dataset class
    class Mol_TestDataset(Dataset):
        '''
        Create the Pytorch Test dataset
        '''
        def __init__(self):
            # Initialize data
            self.n_samples = len(y_test)

            # Select the x data
            self.x_data = torch.from_numpy(x_test) # size [n_samples, n_features]
            self.x_data = self.x_data.to(torch.float32)

            # Select the y data
            self.y_data = torch.from_numpy(y_test) # size [n_samples, 1]
            self.y_data = self.y_data.to(torch.float32)

        # Indexing
        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        # len(dataset) to return the size
        def __len__(self):
            return self.n_samples

    #================================ Dataset Loading =====================================
    # train_dataset
    train_dataset = Mol_TrainDataset()
    # Val_dataset
    val_dataset = Mol_ValDataset()
    # test_dataset
    test_dataset = Mol_TestDataset()

    return train_dataset, val_dataset, test_dataset