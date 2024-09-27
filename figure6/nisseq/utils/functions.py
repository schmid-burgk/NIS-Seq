import numpy.typing as npt
import torch


def arr_to_tensor(x: npt.NDArray) -> torch.Tensor:
    return torch.tensor(x)
