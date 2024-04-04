import torch.nn as nn

class Net_1Hlayer(nn.Module):
    '''
    Neural Network with 1 hidden layer

    Parameters
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (1 element)
        Tuple with the number of neurons in the hidden layer (n)

    activation_functions : `List` `str` (2 elements)
        List with the modules for each activation function in string 
        format ['nn.ReLU()', ..., n]. If last element equals to 'None',
        not af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_1Hlayer, self).__init__()

        hidden_size1 = dimension[0]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, n_targets)
        self.af2 = eval(activation_functions[1])

    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x)) if self.af2 != None else self.linear2(x)

        return x

class Net_2Hlayer(nn.Module):
    '''
    Neural Network with 2 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (2 elements)
        Tuple with the number of neurons in each hidden layer (3, 6)

    activation_functions : `List` `str` (3 elements)
        List with the modules for each activation function in string 
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_2Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        
        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, n_targets)
        self.af3 = eval(activation_functions[2])

    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x)) if self.af3 != None else self.linear3(x)
        
        return x

class Net_3Hlayer(nn.Module):
    '''
    Neural Network with 3 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (3 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9)

    activation_functions : `List` `str` (4 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_3Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)        
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, n_targets)
        self.af4 = eval(activation_functions[3])

    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x)) if self.af4 != None else self.linear4(x)
        
        return x

class Net_4Hlayer(nn.Module):
    '''
    Neural Network with 4 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (4 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12)

    activation_functions : `List` `str` (5 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_4Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, n_targets)
        self.af5 = eval(activation_functions[4])
        
    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x)) if self.af5 != None else self.linear5(x)
        
        return x

class Net_5Hlayer(nn.Module):
    '''
    Neural Network with 5 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (5 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12)

    activation_functions : `List` `str` (6 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_5Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, n_targets)
        self.af6 = eval(activation_functions[5])
        
    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x)) if self.af6 != None else self.linear6(x)
        
        return x

class Net_6Hlayer(nn.Module):
    '''
    Neural Network with 6 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (6 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12, 1)

    activation_functions : `List` `str` (7 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_6Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]
        hidden_size6 = dimension[5]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, hidden_size6)
        self.af6 = eval(activation_functions[5])
        self.linear7 = nn.Linear(hidden_size6, n_targets)
        self.af7 = eval(activation_functions[6])
        
    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x))
        x = self.af7(self.linear7(x)) if self.af7 != None else self.linear7(x)

        return x
    
class Net_7Hlayer(nn.Module):
    '''
    Neural Network with 7 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (6 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12, 1)

    activation_functions : `List` `str` (7 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_7Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]
        hidden_size6 = dimension[5]
        hidden_size7 = dimension[6]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, hidden_size6)
        self.af6 = eval(activation_functions[5])
        self.linear7 = nn.Linear(hidden_size6, hidden_size7)
        self.af7 = eval(activation_functions[6])
        self.linear8 = nn.Linear(hidden_size7, n_targets)
        self.af8 = eval(activation_functions[7])
        
    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x))
        x = self.af7(self.linear7(x))
        x = self.af8(self.linear8(x)) if self.af8 != None else self.linear8(x)

        return x
    
class Net_8Hlayer(nn.Module):
    '''
    Neural Network with 8 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (6 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12, 1)

    activation_functions : `List` `str` (7 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_8Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]
        hidden_size6 = dimension[5]
        hidden_size7 = dimension[6]
        hidden_size8 = dimension[7]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, hidden_size6)
        self.af6 = eval(activation_functions[5])
        self.linear7 = nn.Linear(hidden_size6, hidden_size7)
        self.af7 = eval(activation_functions[6])
        self.linear8 = nn.Linear(hidden_size7, hidden_size8)
        self.af8 = eval(activation_functions[7])
        self.linear9 = nn.Linear(hidden_size8, n_targets)
        self.af9 = eval(activation_functions[8])
        
    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x))
        x = self.af7(self.linear7(x))
        x = self.af8(self.linear8(x))
        x = self.af9(self.linear9(x)) if self.af9 != None else self.linear9(x)

        return x
    
class Net_9Hlayer(nn.Module):
    '''
    Neural Network with 9 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (6 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12, 1)

    activation_functions : `List` `str` (7 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_9Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]
        hidden_size6 = dimension[5]
        hidden_size7 = dimension[6]
        hidden_size8 = dimension[7]
        hidden_size9 = dimension[8]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, hidden_size6)
        self.af6 = eval(activation_functions[5])
        self.linear7 = nn.Linear(hidden_size6, hidden_size7)
        self.af7 = eval(activation_functions[6])
        self.linear8 = nn.Linear(hidden_size7, hidden_size8)
        self.af8 = eval(activation_functions[7])
        self.linear9 = nn.Linear(hidden_size8, hidden_size9)
        self.af9 = eval(activation_functions[8])
        self.linear10 = nn.Linear(hidden_size9, n_targets)
        self.af10 = eval(activation_functions[9])
        
    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x))
        x = self.af7(self.linear7(x))
        x = self.af8(self.linear8(x))
        x = self.af9(self.linear9(x))
        x = self.af10(self.linear10(x)) if self.af10 != None else self.linear10(x)

        return x
    
class Net_10Hlayer(nn.Module):
    '''
    Neural Network with 10 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (6 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12, 1)

    activation_functions : `List` `str` (7 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_10Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]
        hidden_size6 = dimension[5]
        hidden_size7 = dimension[6]
        hidden_size8 = dimension[7]
        hidden_size9 = dimension[8]
        hidden_size10 = dimension[9]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, hidden_size6)
        self.af6 = eval(activation_functions[5])
        self.linear7 = nn.Linear(hidden_size6, hidden_size7)
        self.af7 = eval(activation_functions[6])
        self.linear8 = nn.Linear(hidden_size7, hidden_size8)
        self.af8 = eval(activation_functions[7])
        self.linear9 = nn.Linear(hidden_size8, hidden_size9)
        self.af9 = eval(activation_functions[8])
        self.linear10 = nn.Linear(hidden_size9, hidden_size10)
        self.af10 = eval(activation_functions[9])
        self.linear11 = nn.Linear(hidden_size10, n_targets)
        self.af11 = eval(activation_functions[10])

    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x))
        x = self.af7(self.linear7(x))
        x = self.af8(self.linear8(x))
        x = self.af9(self.linear9(x))
        x = self.af10(self.linear10(x))
        x = self.af11(self.linear11(x)) if self.af11 != None else self.linear11(x)

        return x
    
class Net_11Hlayer(nn.Module):
    '''
    Neural Network with 11 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (6 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12, 1)

    activation_funcif self.af11 != None else self.linear11(x)tions : `List` `str` (7 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_11Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]
        hidden_size6 = dimension[5]
        hidden_size7 = dimension[6]
        hidden_size8 = dimension[7]
        hidden_size9 = dimension[8]
        hidden_size10 = dimension[9]
        hidden_size11 = dimension[10]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, hidden_size6)
        self.af6 = eval(activation_functions[5])
        self.linear7 = nn.Linear(hidden_size6, hidden_size7)
        self.af7 = eval(activation_functions[6])
        self.linear8 = nn.Linear(hidden_size7, hidden_size8)
        self.af8 = eval(activation_functions[7])
        self.linear9 = nn.Linear(hidden_size8, hidden_size9)
        self.af9 = eval(activation_functions[8])
        self.linear10 = nn.Linear(hidden_size9, hidden_size10)
        self.af10 = eval(activation_functions[9])
        self.linear11 = nn.Linear(hidden_size10, hidden_size11)
        self.af11 = eval(activation_functions[10])
        self.linear12 = nn.Linear(hidden_size11, n_targets)
        self.af12 = eval(activation_functions[11])

    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x))
        x = self.af7(self.linear7(x))
        x = self.af8(self.linear8(x))
        x = self.af9(self.linear9(x))
        x = self.af10(self.linear10(x))
        x = self.af11(self.linear11(x))
        x = self.af12(self.linear12(x)) if self.af12 != None else self.linear12(x)

        return x
    
class Net_12Hlayer(nn.Module):
    '''
    Neural Network with 12 hidden layers

    Attributes
    ----------
    n_features : `int`
        Number of parameters given in the model input

    n_targets : `int`
        Number of targets expected in output layer

    dimension : `tuple` `int` (6 elements)
        Tuple with the number of neurons in each hidden layer (3, 6, 9, 12, 1)

    activation_funcif self.af11 != None else self.linear11(x)tions : `List` `str` (7 elements)
        List with the modules for each activation function in string
        format ['nn.ReLU()', ..., n]. If last element equals to 'None', not 
        af will be applied to the output.
    '''
    def __init__(self, n_features, n_targets, dimension, activation_functions):
        super(Net_12Hlayer, self).__init__()

        hidden_size1 = dimension[0]
        hidden_size2 = dimension[1]
        hidden_size3 = dimension[2]
        hidden_size4 = dimension[3]
        hidden_size5 = dimension[4]
        hidden_size6 = dimension[5]
        hidden_size7 = dimension[6]
        hidden_size8 = dimension[7]
        hidden_size9 = dimension[8]
        hidden_size10 = dimension[9]
        hidden_size11 = dimension[10]
        hidden_size12 = dimension[11]

        self.linear1 = nn.Linear(n_features, hidden_size1)
        self.af1 = eval(activation_functions[0])
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.af2 = eval(activation_functions[1])
        self.linear3 = nn.Linear(hidden_size2, hidden_size3)
        self.af3 = eval(activation_functions[2])
        self.linear4 = nn.Linear(hidden_size3, hidden_size4)
        self.af4 = eval(activation_functions[3])
        self.linear5 = nn.Linear(hidden_size4, hidden_size5)
        self.af5 = eval(activation_functions[4])
        self.linear6 = nn.Linear(hidden_size5, hidden_size6)
        self.af6 = eval(activation_functions[5])
        self.linear7 = nn.Linear(hidden_size6, hidden_size7)
        self.af7 = eval(activation_functions[6])
        self.linear8 = nn.Linear(hidden_size7, hidden_size8)
        self.af8 = eval(activation_functions[7])
        self.linear9 = nn.Linear(hidden_size8, hidden_size9)
        self.af9 = eval(activation_functions[8])
        self.linear10 = nn.Linear(hidden_size9, hidden_size10)
        self.af10 = eval(activation_functions[9])
        self.linear11 = nn.Linear(hidden_size10, hidden_size11)
        self.af11 = eval(activation_functions[10])
        self.linear12 = nn.Linear(hidden_size11, hidden_size12)
        self.af12 = eval(activation_functions[11])
        self.linear13 = nn.Linear(hidden_size12, n_targets)
        self.af13 = eval(activation_functions[12])

    def forward(self, x):
        x = self.af1(self.linear1(x))
        x = self.af2(self.linear2(x))
        x = self.af3(self.linear3(x))
        x = self.af4(self.linear4(x))
        x = self.af5(self.linear5(x))
        x = self.af6(self.linear6(x))
        x = self.af7(self.linear7(x))
        x = self.af8(self.linear8(x))
        x = self.af9(self.linear9(x))
        x = self.af10(self.linear10(x))
        x = self.af11(self.linear11(x))
        x = self.af12(self.linear12(x))
        x = self.af13(self.linear13(x)) if self.af13 != None else self.linear13(x)

        return x