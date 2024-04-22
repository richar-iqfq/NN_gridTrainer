import itertools
import random
import string

import numpy as np

#=================== For training =========================
def get_around_learning_rates(fixed_decimal, increments, exponential_base, max_exponents):
    '''
        fixed_decimal = 0.01     # lowest limit = 0.0001. IF CERO, increments over the whole the loop is by 0.1 per 0.1.
                                        # which is convinient for the preliminar analisis of thelearning rate. If different
                                        # than zero, it takes values around the indicated value.
        increments = 100           #amount of increments to the fixed decimal, of the same order of the fixed decimal

        exponential_base = 5      # the number that will be added in the position acordingly to the exponential. Must be higher
                                # or equal than 1
        max_exponents = 4
    '''
    # this routine is only for proper rounding and for the generation of the numbers
    r1 = str(exponential_base)
    r1 =int(r1[::-1].find('.'))
    r2 = str(fixed_decimal)
    r2 =int(r2[::-1].find('.'))
    r3 = r2

    lr_list = []

    # r2 is less than zero for fixed_decimal values less than 0.0001. In those cases the generated lr may be negative or
    # higher than one. This condition prevents this situation.
    if r2 < 0:
        r2 = 1
    # Setting the proper rounding of the generated lr
    if r2 > r1:
        rf = r2
    else:
        rf = r1

    for s in range(0, increments):
        s = fixed_decimal + s * 10 ** (-r2)
        lr = round(s, 6)
        if lr == 0:
            lr = 0.000001
        lr_list.append(lr)
        # print(lr, '*/*/*/')
        for exponent in range(1, max_exponents):
            add = exponential_base * 10 ** - (exponent + r2)
            lr = round(s + add, exponent + rf)
            lr_list.append(lr)
            # print(lr, '*****')
        if s == fixed_decimal and r3 > 0:           # for the first loop, also explores lower values than the fixed_decimal
            for exponent in range(1, max_exponents):
                add = exponential_base * 10 ** - (exponent + r2)
                lr = round(s - add, exponent + rf)
                lr_list.append(lr)
                # print(lr,'__+++___')                # values that will be consiered (lower than fixed decimal)
    
    learning_rates = np.array(lr_list)
    filter = learning_rates < 1

    return learning_rates[filter]

def get_adjacent_batches(batch_size: int) -> tuple:
    values = [16, 32, 64, 128, 256, 512, 1024, 2048]

    diff = 2048
    k = 0

    for i in range(len(values) - 1):
        j = i + 1

        compute = abs(batch_size - values[i]) + abs(batch_size - values[j])

        if compute <= diff:
            diff = compute
            k = i

    return values[k], values[k+1]

def get_random_activation_functions(af_list: list, n_functions: int, lineal_output: bool, seed: int=None) -> list:
    '''
    Return a list with all the possible combinations of `n` activation functions
    readed from `parameters`.

    Parameters
    ----------
    n_functions (`int`):
        Number of activation functions needed.
    
    oN (`bool`):
        if True the last activation function will set to None. Deafaul is False
    
    seed (`int` or `None`):
        if a value is given, will set a random seed to get the random values
        
    Returns
    -------
    P (`list` of `str`): 
        List with the activation functions
    '''
    af_valid = af_list

    if lineal_output:
        P = [p for p in itertools.product(af_valid, repeat=n_functions-1)] # Too many choices
    else:
        P = [p for p in itertools.product(af_valid, repeat=n_functions)] # Too many choices
    
    AF_combinations = []

    rnd = np.random.RandomState(seed=seed)
    index_list = rnd.randint(0, len(P), 50)

    for index in index_list:
        AF = P[index]
        if lineal_output:
            AF = list(AF)
            AF.append('None')
            AF = tuple(AF)

        AF_combinations.append(AF)

    return AF_combinations

def get_random_layer_sizes(hidden_layers: int, max_neurons: int, seed: int=None) -> list:
    '''
    Return a list with all the possible combinations of `n` hidden layers and max `m` neurons
    of each layer from the values `[4,6,8,10,12,...,m]`.

    Parameters
    ----------
    hidden_layers (`int`):
        Number of hidden layers.
    
    Returns
    -------
    P `list` `int`: List with the number of neurons for each layer combinations. 
    '''
    # values1 = [i for i in range(4, neurons+1) if i % 2 == 0]
    # values2 = [i for i in range(3, neurons+1) if i % 2 != 0]

    values = [i for i in range(3, max_neurons+1)]

    T = [p for p in itertools.product(values, repeat=hidden_layers)] # Too many choices

    LY_combinations = []

    rnd = np.random.RandomState(seed=seed)
    index_list = rnd.randint(0, len(T), 30)

    for index in index_list:
        LY_combinations.append(T[index])

    return LY_combinations

def generate_random_string(n: int) -> str:
    letters_list = random.choices(string.ascii_letters, k=25)

    return ''.join(letters_list)

def is_model_stuck(acc_list: list, r2_list: list) -> bool:
    flag = False

    if np.mean(acc_list[-15::]) == 0:
        flag = True
    
    if np.mean(r2_list[-5::]) == 0:
        flag = True
    
    return flag