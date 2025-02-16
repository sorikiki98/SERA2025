import enum
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NeTIBatch:
    input_ids: torch.Tensor
    input_ids_placeholder_img: torch.Tensor
    input_ids_placeholder_new: torch.Tensor
    image_embeds: torch.Tensor
    timesteps: torch.Tensor
    unet_layers: torch.Tensor
    truncation_idx: Optional[int] = None


@dataclass
class PESigmas:
    sigma_t: float
    sigma_l: float


@dataclass
class MapperOutput:
    word_embedding: torch.Tensor
    bypass_output: torch.Tensor
    bypass_unconstrained: bool
    output_bypass: bool
    output_bypass_alpha: float
