import torch
import numpy as np
import torch

#=================== For array manipulation ===============
def move_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    # PyTorch has poor support for half tensors (float16) on CPU.
    if tensor.dtype in {torch.bfloat16, torch.float16}:
        tensor = tensor.to(dtype=torch.float32)

    return tensor.cpu()

def convert_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy()

def convert_to_torch(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    return tensor.to(torch.float32)

def process_tensor(tensor: torch.Tensor) -> np.ndarray:
    if torch.is_tensor(tensor):
        tensor = move_to_cpu(tensor)
        tensor = convert_to_numpy(tensor)

    return tensor

def process_array(array: np.ndarray) -> torch.Tensor:
    if isinstance(array, np.ndarray):
        array = convert_to_torch(array)
        array = move_to_cpu(array)

    return array

def is_upper_lower_artifact(y_pred: np.ndarray, n_targets: int) -> bool:
    epsilon = 0.003
    flag = False
    tol = 10

    # Move to cpu and convert to np.ndarray
    y_pred = process_tensor(y_pred)

    for i in range(n_targets):
        # Select target
        y_target = y_pred[:, i]

        # Get max and min values
        y_max = np.max(y_target)
        y_min = np.min(y_target)

        # Get upper and lower values
        y_up = y_target >= (y_max - epsilon)
        y_lo = y_target <= (y_min + epsilon)

        # Get count
        count_up = np.count_nonzero(y_up)
        count_lo = np.count_nonzero(y_lo)

        if count_lo > tol or count_up > tol:
            flag = True

    return flag

