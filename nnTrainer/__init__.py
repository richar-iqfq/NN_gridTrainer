__coder__ = 'Ricardo (richariqfq)'
__version__ = '1.0.0'

from nnTrainer.config.Configurator import Configurator

# ---------------- Constants ------------------
# Set training steps
Grid = 'Grid'
Optimization = 'Optimization'
RandomState = 'RandomState'
TuningBatch = 'TuningBatch'
TuningLr = 'TuningLr'
Lineal = 'Lineal'
AroundExploration = 'AroundExploration'
RestartGridFromWorst = 'RestartGridFromWorst'
Testing = 'Testing'
AddLayers = 'AddLayers'

# Path by step
path_name = {
    'Grid' : 'Grid',
    'Optimization' : 'Optimization',
    'TuningBatch' : 'TuningBatch',
    'TuningLr' : 'TuningLr',
    'Lineal' : 'Lineal',
    'RandomState' : 'RandomState',
    'AroundExploration' : 'AroundExploration',
    'Recovering' : 'Recovering',
    'RestartGridFromWorst' : 'Grid',
    'Testing' : 'Test',
}

# Instance singleton configurator
config = Configurator()