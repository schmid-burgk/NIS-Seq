import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def eval_model_on_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, npt.NDArray]:
    model.eval()
    model.to(device)

    outputs_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader), desc="Eval: "):
            inputs: torch.Tensor
            labels: torch.Tensor

            inputs, labels = inputs.to(device), labels.to(device)

            outputs: torch.Tensor = model(inputs)
            outputs_list.append(outputs.cpu().detach().numpy())
            labels_list.append(labels.cpu().detach().numpy())

    return {
        "outputs": np.concatenate(outputs_list),
        "labels": np.concatenate(labels_list),
    }
