import random
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from transformers import CLIPVisionModel

from constants import UNET_LAYERS
from models.positional_encoding import NeTIPositionalEncoding, BasicEncoder
from utils.types import PESigmas, MapperOutput


class NeTIMapper(nn.Module):
    """ Main logic of our NeTI mapper. """

    def __init__(
            self,
            embedding_type: Literal['img'],
            output_dim: int = 768,
            token_embed_dim: int = 768,
            unet_layers: List[str] = UNET_LAYERS,
            arch_mlp_hidden_dims: int = 128,
            use_nested_dropout: bool = True,
            nested_dropout_prob: float = 0.5,
            norm_scale: Optional[torch.Tensor] = None,
            use_positional_encoding=1,
            num_pe_time_anchors: int = 10,
            pe_sigmas: PESigmas = PESigmas(sigma_t=0.03,
                                           sigma_l=2.0),
            output_bypass: bool = True,
            arch_disable_tl: bool = True,
            original_ti_init_embed=None,
            original_ti: bool = False,
            bypass_unconstrained: bool = True,
            output_bypass_alpha: float = 0.2,
            placeholder_token: str = None,
    ):
        """
        Args:
        embedding_type: whether the Neti-mapper should learn object or view
            control. View-control will condition on camera pose as well. MLP
            architecture is also different.
        placeholder_view_tokens: all possible view_tokens used for training.
            Ignored if embedding_type=='object'.
        placeholder_view_tokens_ids: token ids for `placeholder_view_tokens`
        arch_view_disable_tl: do not condition on timestep and unet layer (t,l)
        original_ti: run the 'original TI'
        bypass_unconstrained: passed through in the output
        """
        super().__init__()
        self.embedding_type = embedding_type
        self.token_embed_dim = token_embed_dim
        self.use_nested_dropout = use_nested_dropout
        self.nested_dropout_prob = nested_dropout_prob
        self.arch_mlp_hidden_dims = arch_mlp_hidden_dims
        self.norm_scale = norm_scale
        self.original_ti = original_ti
        self.arch_disable_tl = arch_disable_tl
        self.original_ti_init_embed = original_ti_init_embed
        self.output_bypass = output_bypass
        self.output_bypass_alpha = output_bypass_alpha
        self.num_unet_layers = len(unet_layers)
        self.placeholder_token = placeholder_token  # does nothing
        self.bypass_unconstrained = bypass_unconstrained

        if original_ti and output_bypass:
            raise ValueError(
                f"If doing cfg.model.original_ti=[True]",
                f" then you cannot have cfg.model.original_ti=[True]")
        self.output_bypass = output_bypass
        if self.output_bypass:
            output_dim *= 2  # Output two vectors

        # use the legacy (t,l) conditioning. For later exps, call func for setup
        self.pe_sigmas = pe_sigmas
        self.use_positional_encoding = use_positional_encoding
        if type(self.use_positional_encoding) is bool:
            self.use_positional_encoding = int(
                self.use_positional_encoding)
        if self.use_positional_encoding == 1:
            self.encoder = NeTIPositionalEncoding(
                sigma_t=pe_sigmas.sigma_t,
                sigma_l=pe_sigmas.sigma_l).cuda()
            self.input_dim = num_pe_time_anchors * len(unet_layers)
        elif self.use_positional_encoding == 0:
            self.encoder = BasicEncoder().cuda()
            self.input_dim = 2
        elif self.use_positional_encoding == 2:
            raise NotImplementedError()
        else:
            raise ValueError()

        self.input_layer = self.set_input_layer(len(unet_layers),
                                                num_pe_time_anchors)

        # define architecture
        if self.embedding_type == "img":
            self.set_net_image(num_unet_layers=len(unet_layers),
                               num_time_anchors=num_pe_time_anchors,
                               output_dim=output_dim)
        else:
            raise ValueError(f"Unknown embedding type {self.embedding_type}")

    def set_net_image(self,
                      num_unet_layers: int,
                      num_time_anchors: int,
                      output_dim: int = 768):
        if self.original_ti:
            self.ti_embeddings = torch.nn.parameter.Parameter(
                self.original_ti_init_embed.unsqueeze(0), requires_grad=True)
            self.output_layer = nn.Identity()

        else:
            h_dim = self.arch_mlp_hidden_dims
            self.net = nn.Sequential(nn.Linear(self.input_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(h_dim, output_dim))

    def set_input_layer(self, num_unet_layers: int,
                        num_time_anchors: int) -> nn.Module:
        if self.use_positional_encoding:
            input_layer = nn.Linear(self.encoder.num_w * 3, self.input_dim)
            input_layer.weight.data = self.encoder.init_layer(
                num_time_anchors, num_unet_layers)
        else:
            input_layer = nn.Identity()
        return input_layer

    def forward(self,
                timestep: torch.Tensor,
                unet_layer: torch.Tensor,
                image_embeds: torch.Tensor,
                truncation_idx: int = None,
                ) -> MapperOutput:
        """
        Args:
        input_ids_placeholder_view: If embedding_type=='object', ignored. If
            embedding_type=='view', use the token id to condition on the view
            parameters for that token.
        """
        embedding = self.extract_hidden_representation(
            timestep, unet_layer, image_embeds)  # [bs, 128]

        if self.use_nested_dropout:
            embedding = self.apply_nested_dropout(
                embedding, truncation_idx=truncation_idx)

        output = self.get_output(embedding)  # [bs, 1536]

        return output

    def get_encoded_input(self, timestep: torch.Tensor,
                          unet_layer: torch.Tensor,
                          image_embeds: torch.Tensor) -> torch.Tensor:
        """ Encode the (t,l, image_token_embeds) params """
        encoded_input = self.encoder.encode(
            timestep,
            unet_layer,
            image_embeds
        )  # [bs, 2304]

        return self.input_layer(encoded_input)  # (bs, 160)

    def extract_hidden_representation(
            self, timestep: torch.Tensor, unet_layer: torch.Tensor
            , image_embeds: torch.Tensor) -> torch.Tensor:
        encoded_input = self.get_encoded_input(timestep, unet_layer, image_embeds)  # [bs, 160]
        embedding = self.net(encoded_input)  # [bs, 128]

        return embedding

    def apply_nested_dropout(self,
                             embedding: torch.Tensor,
                             truncation_idx: int = None) -> torch.Tensor:
        if self.training:
            if random.random() < self.nested_dropout_prob:
                dropout_idxs = torch.randint(low=0,
                                             high=embedding.shape[1],
                                             size=(embedding.shape[0],))
                for idx in torch.arange(embedding.shape[0]):
                    embedding[idx][dropout_idxs[idx]:] = 0
        if not self.training and truncation_idx is not None:
            for idx in torch.arange(embedding.shape[0]):
                embedding[idx][truncation_idx:] = 0
        return embedding

    def get_output(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding = self.output_layer(embedding)

        # split word embedding and output bypass (if enabled) and save to object
        if not self.output_bypass:
            output = MapperOutput(word_embedding=embedding,
                                  bypass_output=None,
                                  bypass_unconstrained=False,
                                  output_bypass=self.output_bypass,
                                  output_bypass_alpha=self.output_bypass_alpha)
        else:
            dim = embedding.shape[1] // 2
            output = MapperOutput(
                word_embedding=embedding[:, :dim],
                bypass_output=embedding[:, dim:],
                bypass_unconstrained=self.bypass_unconstrained,
                output_bypass=self.output_bypass,
                output_bypass_alpha=self.output_bypass_alpha)

        # apply norm scaling to the word embedding (if enabled)
        if self.norm_scale is not None:
            output.word_embedding = F.normalize(output.word_embedding,
                                                dim=-1) * self.norm_scale

        return output
