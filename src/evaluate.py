from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import lightning as L
import numpy as np
from rich.console import Console
import torch

from src.datasets.datamodule import Datamodule

CONSOLE = Console(width=170)

torch.set_float32_matmul_precision("medium")


def to_native(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    native_dict = {}
    for field_name, field in input_dict.items():
        if isinstance(field[0], torch.Tensor):
            native_dict[field_name] = torch.cat(field, dim=0)
        elif isinstance(field[0], list):
            native_dict[field_name] = sum(field, [])
    return native_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    OmegaConf.register_new_resolver("eval", eval)

    assert config.checkpoint_path is not None, "Checkpoint path must be provided."

    trainer = L.Trainer(
        accelerator="cuda",
        strategy="auto",
        devices=1,
        num_sanity_val_steps=0,
    )

    model = instantiate(config.model)

    dataset = instantiate(config.dataset)
    datamodule = Datamodule(
        train_dataset=deepcopy(dataset).set_split("train"),
        eval_dataset=deepcopy(dataset).set_split("test"),
        batch_train_size=config.batch_size,
        num_workers=8,
    )

    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.checkpoint_path)
    metrics_dict = model.test_contrastive_metrics

    ckpt_name = Path(config.checkpoint_path).stem
    metric_filename = Path(config.log_dir).parent / f"{ckpt_name}-metrics.npy"
    metric_filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(metric_filename, metrics_dict)

    CONSOLE.print(f"Metrics saved at {metric_filename}.")


if __name__ == "__main__":
    main()
