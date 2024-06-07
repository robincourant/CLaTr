from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import lightning as L
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

    out = trainer.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.checkpoint_path,
        return_predictions=True,
    )

    preds = defaultdict(list)
    for batch in out:
        preds["filenames"].append([x[:-4] for x in batch["traj_filename"]])
        preds["masks"].append(batch["padding_mask"])
        preds["caption"].append(batch["caption_raw"]["caption"])
        preds["token"].append(batch["caption_raw"]["token"])

        preds["ref_matrices"].append(batch["ref_matrices"])
        preds["t_matrices"].append(batch["t_matrices"])
        preds["m_matrices"].append(batch["m_matrices"])
        preds["t_latents"].append(batch["t_latents"])
        preds["m_latents"].append(batch["m_latents"])

    preds = to_native(preds)

    ckpt_name = Path(config.checkpoint_path).stem
    pred_filename = Path(config.log_dir).parent / f"{ckpt_name}-preds.npy"
    pred_filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_filename, preds)

    CONSOLE.print(f"Predictions saved at {pred_filename}.")


if __name__ == "__main__":
    main()
