import numpy as np

def get_learning_rates(fixed_decimal, increments, exponential_base, max_exponents):
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

lr = get_learning_rates(0.1, 100, 5, 2)

for l in lr:
    print(l)