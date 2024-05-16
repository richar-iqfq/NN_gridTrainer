network = [
    {
        "hidden_layers" : 5,
        "dimension" : (12, 13, 14, 15, 6),
        "activation_functions" : (
            'nn.LeakyReLU()',
            'nn.Tanhshrink()',
            'nn.SELU()',
            'nn.ReLU()',
            'nn.ReLU()',
            'None'
        ),
        "optimizer" : "Adam",
        "criterion" : 'nn.L1Loss()',
        "random_state" : 12356,
        "num_epochs" : 800,
        "batch_size" : 31,
        "lr" : 0.001,
    }
]