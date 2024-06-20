from ..base_class.Module import Module
from ..base_class.Launcher import MainLauncher

class DissociationPlot(Module, ):
    def __init__(self) -> None:
        super().__init__()
    
    def after(self, step: str) -> bool:
        launcher = MainLauncher()

        lower_hidden_size = launcher.start_point
        upper_hidden_size = launcher.max_hidden_layers

        for hidden in range(lower_hidden_size, upper_hidden_size):
            better_network = launcher.recover_network(hidden, step)

            for network in better_network:
                path = network['Path']

                model = 