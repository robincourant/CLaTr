from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils.file_utils import load_txt


class CaptionDataset(Dataset):
    def __init__(
        self,
        name: str,
        dataset_dir: str,
        num_cams: int,
        lm: Dict[str, Any],
        num_segments: int,
        **kwargs,
    ):
        super().__init__()
        self.modality = name
        self.name = name
        self.dataset_dir = Path(dataset_dir)
        # Set data paths (segments, captions, etc...)
        for name, field in kwargs.items():
            if isinstance(field, str):
                field = Path(field)
            setattr(self, name, field)

        self.filenames = None

        for name, field in lm.items():
            setattr(self, name, field)

        self.num_cams = num_cams
        self.num_segments = num_segments

        # For caption filtering (based on CLIP token similarity)
        self.clip_token_dir = self.dataset_dir / "caption_cam_clip" / "token"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        # Load data
        if hasattr(self, "segment_dir"):
            raw_segments = torch.from_numpy(
                np.load((self.segment_dir / (filename + ".npy")))
            )
            padded_raw_segments = F.pad(
                raw_segments,
                (0, self.num_cams - len(raw_segments)),
                value=self.num_segments,
            )
        if hasattr(self, "raw_caption_dir"):
            raw_caption = load_txt(self.raw_caption_dir / (filename + ".txt"))
        if hasattr(self, "feat_caption_dir"):
            feat_caption = torch.from_numpy(
                np.load((self.feat_caption_dir / "seq" / (filename + ".npy")))
            )
            padded_feat_caption = F.pad(
                feat_caption.to(torch.float32),
                (0, 0, 0, self.max_feat_length - feat_caption.shape[0]),
            )
            feat_mask = torch.ones((self.max_feat_length))
            feat_mask[feat_caption.shape[0] :] = 0

            token_caption = torch.from_numpy(
                np.load(self.clip_token_dir / (filename + ".npy"))
            )

        if self.modality == "caption":
            raw_data = {
                "caption": raw_caption,
                "segments": padded_raw_segments,
                "token": token_caption,
            }
            feat_data = {"x": padded_feat_caption, "mask": feat_mask.to(bool)}

        elif self.modality == "segments":
            raw_data = {"segments": padded_raw_segments}
            # Shift by one for padding
            feat_data = F.one_hot(
                padded_raw_segments, num_classes=self.num_segments + 1
            ).to(torch.float32)
            feat_data = feat_data

        else:
            raise ValueError(f"Modality {self.modality} not supported")

        return filename, feat_data, raw_data
