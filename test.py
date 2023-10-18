from modules.Models import Net_1Hlayer
from modules.Configurator import Configurator
from modules.PreprocessData import PreprocessData
import torch
import torch.nn as nn

config = Configurator()
config.update(
    config_file='config.json'
)

preprocess = PreprocessData(config)

ID, x, y = preprocess.Retrieve_Processed()

network = Net_1Hlayer(config.json['num_features'], config.json['num_targets'], (2,), ("nn.ELU()","nn.LeakyReLU()"), config.json['af_valid'])

inputx = torch.from_numpy(x)
inputx = inputx.to(torch.float32)

outputy = torch.from_numpy(y)
outputy = outputy.to(torch.float32)

optimizer = torch.optim.AdamW(network.parameters(), lr=0.001)
criterion = nn.L1Loss()

for epoch in range(1500):
    optimizer.zero_grad()

    y_pred = network(inputx)
    loss = criterion(y_pred, outputy)

    loss.backward()
    optimizer.step()

    if (epoch-1)%10 == 0:
        print(f'epoch: {epoch}     loss: {loss.item()}')

print(outputy)
print(y_pred)