__coder__ = 'Ricardo (richariqfq)'
__version__ = '1.0.0'

# ---- Constants ----

# Set training steps
grid = 'grid'
optimization = 'optimization'
random_state = 'random_state'
tuning_batch = 'tuning_batch'
tuning_lr = 'tuning_lr'
lineal = 'lineal'
around_exploration = 'around_exploration'
restart_grid_from_worst = 'restart_grid_from_worst'

from nnTrainer.config.Configurator import Configurator

# Instance singleton configurator
config = Configurator()

from .launch.Launcher import Launcher