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