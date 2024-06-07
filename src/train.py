from copy import deepcopy
from typing import Optional

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch

from src.datasets.datamodule import Datamodule
from utils.random_utils import set_random_seed

torch.set_float32_matmul_precision("medium")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    OmegaConf.register_new_resolver("eval", eval)

    set_random_seed(config.seed)

    if config.log_wandb:
        logger = WandbLogger(
            entity=config.entity,
            project=config.project_name,
            name=config.xp_name,
            save_dir=config.log_dir,
        )
    else:
        logger = None

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        every_n_epochs=1,
        save_top_k=-1,
        verbose=True,
    )

    trainer = L.Trainer(
        accelerator="cuda",
        strategy="auto",
        devices=1,
        max_epochs=config.num_train_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
    )

    model = instantiate(config.model)

    dataset = instantiate(config.dataset)
    datamodule = Datamodule(
        deepcopy(dataset).set_split("train"),
        deepcopy(dataset).set_split("val"),
        config.batch_size,
        config.num_workers,
    )
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.checkpoint_path)


if __name__ == "__main__":
    main()
