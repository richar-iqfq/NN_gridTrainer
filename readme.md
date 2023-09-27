# NN_gridTrainer

- Ambriz Medina Ricardo

Repositorio dedicado al entrenamiento de redes neuronales mediante la estrategia de mallado y ensamblaje de capas ocultas. El proceso consiste en los siguientes pasos:

## Estrategia de búsqueda

- grid: En esta etapa se realiza un recorrido sobre el numero de neuronas dentro del rango especificado y la función de activación de cada capa.
- optimization: En esta etapa se prueban las distintas funciones de perdida y optimizadores para la arquitectura encontrada previamente.
- tuning_lr: Se prueba un conjunto de valores de learning rate para la arquitectura encontrada previamente.
- tuning_batch: Se prueba un conjunto de tamaño de lote para la arquitectura encontrada previamente.
- lineal: Se trata al proceso de entrenamiento como un modelo lineal y se busca la minimización de la pérdida de la red mediante el algoritmo de Nelder Mead.
- random_state: Se prueban diferentes valores de random state para el split del set de entrenamiento.
- around_exploration: Se realiza una búsqueda alrededor de los valores óptimos encontrados de lr y batch.

## Uso

coming soon...