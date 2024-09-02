"""Code adapted from: https://github.com/Mathux/TMR"""

import os
from omegaconf import DictConfig
from pathlib import Path

import hydra
import torch


# split the lightning checkpoint into
# seperate state_dict modules for faster loading
def extract_ckpt(ckpt_path: Path):
    new_path_template = os.path.join(ckpt_path.parent, "clatr-{}.ckpt")
    ckpt_dict = torch.load(ckpt_path)
    state_dict = ckpt_dict["state_dict"]
    module_names = list(set([x.split(".")[0] for x in state_dict.keys()]))
    # should be ['traj_encoder', 'text_encoder', 'traj_decoder'] for example
    for module_name in module_names:
        path = new_path_template.format(module_name)
        sub_state_dict = {
            ".".join(x.split(".")[1:]): y.cpu()
            for x, y in state_dict.items()
            if x.split(".")[0] == module_name
        }
        torch.save(sub_state_dict, path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def hydra_load_model(cfg: DictConfig) -> None:
    return extract_ckpt(Path(cfg.checkpoint_path))


if __name__ == "__main__":
    hydra_load_model()
