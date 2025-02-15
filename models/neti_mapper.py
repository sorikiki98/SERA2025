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
            placeholder_new_tokens: List[str] = None,
            placeholder_new_token_ids: torch.Tensor = None,
            arch_new_disable_tl: bool = True,
            original_ti_init_embed=None,
            original_ti: bool = False,
            bypass_unconstrained: bool = True,
            output_bypass_alpha: float = 0.2,
            placeholder_new_token: str = None,
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
        self.arch_new_disable_tl = arch_new_disable_tl
        self.original_ti_init_embed = original_ti_init_embed
        self.output_bypass_alpha = output_bypass_alpha
        self.num_unet_layers = len(unet_layers)
        self.placeholder_new_token = placeholder_new_token  # does nothing
        self.bypass_unconstrained = bypass_unconstrained

        if original_ti and output_bypass:
            raise ValueError(
                f"If doing cfg.model.original_ti=[True]",
                f" then you cannot have cfg.model.original_ti=[True]")
        self.output_bypass = output_bypass
        if self.output_bypass:
            output_dim *= 2  # Output two vectors

        # for view mappers, prepare some class properties for the view tokens
        if self.embedding_type == "style":
            '''
            self.placeholder_style_tokens = placeholder_style_tokens
            self.placeholder_style_token_ids = placeholder_style_token_ids
            self._prepare_style_token_param_lookup()
            '''

        # set up positional encoding. For older experiments (arch_view_net<14),
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
        elif self.embedding_type == "style":
            self.set_net_style(num_unet_layers=len(unet_layers),
                               num_time_anchors=num_pe_time_anchors,
                               output_dim=output_dim)

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
                input_ids_placeholder: torch.Tensor,
                image_encoder: CLIPVisionModel,
                truncation_idx: int = None) -> MapperOutput:
        """
        Args:
        input_ids_placeholder_view: If embedding_type=='object', ignored. If
            embedding_type=='view', use the token id to condition on the view
            parameters for that token.
        """
        if self.original_ti or (self.embedding_type == "style"
                                and self.arch_style_net == 1):
            if self.embedding_type == "style":
                idx = [
                    self.lookup_ti_embedding[p.item()].item()
                    for p in input_ids_placeholder
                ]
                embedding = self.ti_embeddings[idx]
            else:
                embedding = self.ti_embeddings[[0] * len(timestep)]
            return embedding

        embedding = self.extract_hidden_representation(
            timestep, unet_layer, input_ids_placeholder, image_encoder)  # [bs, 128]

        if self.use_nested_dropout:
            embedding = self.apply_nested_dropout(
                embedding, truncation_idx=truncation_idx)

        output = self.get_output(embedding)  # [bs, 1536]

        return output

    def get_encoded_input(self, timestep: torch.Tensor,
                          unet_layer: torch.Tensor,
                          input_ids_placeholder: torch.Tensor,
                          image_encoder: CLIPVisionModel) -> torch.Tensor:
        """ Encode the (t,l, style_token_embed) params """
        style_input = tokenizer(input_ids_placeholder)  # [2, 768]
        encoded_input = self.encoder.encode(
            timestep,
            unet_layer,
            style_input
        )  # [bs, 2304]

        return self.input_layer(encoded_input)  # (bs, 160)

    def _prepare_style_token_param_lookup(self):
        """
        All possible view tokens that the mapper handles are known ahead of
        time. This method prepares lookup dicts from tokenids (and tokens) to
        the view parameters.
    
        The `rescale_min_max` thing: at training, we put all the training view
        parameters in the range [-1,1]. If `rescale_min_max=True`, then we define
        the max and min parameters for that normalization based on the current
        self.placeholder_view_tokens. So this should only be called when those
        placeholder tokens are the same as what was done in training.
        For inference, this is handled by the CheckpointHandler.load_mapper()
        method (which calls the public-facing function `add_view_tokens_to_vocab`).
        But if this func is run with `rescale_min_max=True` with the wrong
        placeholders [set] to self, then the generations will become weird
        """
        assert len(self.placeholder_style_tokens) == len(
            self.placeholder_style_token_ids)

        assert all([
            s[:7] == '<style_' for s in self.placeholder_style_tokens
        ]), "not style tokens"

        style_params = [[
            s for s in token[7:]
        ] for token in self.placeholder_style_tokens]

        self.style_token_2_style_params = dict(
            zip(self.placeholder_style_tokens, style_params))
        self.style_tokenid_2_view_params = dict(
            zip(self.placeholder_style_token_ids, style_params))

    def get_style_params_from_token(self, input_ids_placeholder_style, device,
                                    dtype):
        """
        Given a set of token ids, `input_ids_placeholder_view`, that correspond
        to tokens like <"view_10_40_1p2>", return a tensor of the camera params,
        e.g. params (10,40,1.2). The dimension is (bs,3), where each batch element
        is
        """

        # lookup the view parameters
        style_params = [
            self.style_tokenid_2_style_params[i.item()]
            for i in input_ids_placeholder_style
        ]

        return style_params

    def extract_hidden_representation(
            self, timestep: torch.Tensor, unet_layer: torch.Tensor,
            input_ids_placeholder: torch.Tensor,
            image_encoder: CLIPVisionModel) -> torch.Tensor:
        if self.embedding_type == 'object':
            pass
        elif self.embedding_type == 'img':
            encoded_input = self.get_encoded_input(timestep, unet_layer, input_ids_placeholder,
                                                   image_encoder)  # [bs, 160]
            embedding = self.net(encoded_input)  # [bs, 128]
        else:
            raise ValueError()

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
                                  output_bypass_alpha=self.output_bypass_alpha)
        else:
            dim = embedding.shape[1] // 2
            output = MapperOutput(
                word_embedding=embedding[:, :dim],
                bypass_output=embedding[:, dim:],
                bypass_unconstrained=self.bypass_unconstrained,
                output_bypass_alpha=self.output_bypass_alpha)

        # apply norm scaling to the word embedding (if enabled)
        if self.norm_scale is not None:
            output.word_embedding = F.normalize(output.word_embedding,
                                                dim=-1) * self.norm_scale

        return output

    def add_view_tokens_to_vocab(self, placeholder_view_tokens_new: List[str],
                                 placeholder_view_token_ids_new: List[int]):
        """
        Add new tokens to the vocabulary.
        This is intended to be called by external scripts doing inference with
        novel view tokens (tokens that were not used in training).
        """
        assert len(placeholder_view_tokens_new) == len(
            placeholder_view_token_ids_new)

        ## get ids of tokens that are novel
        mask_new_tokens = ~np.isin(np.array(placeholder_view_tokens_new),
                                   np.array(self.placeholder_view_tokens))
        mask_new_token_ids = ~np.isin(
            np.array(placeholder_view_token_ids_new),
            np.array(self.placeholder_view_token_ids))
        assert np.all(np.all(mask_new_tokens == mask_new_token_ids))
        idxs_new_tokens = np.where(mask_new_tokens)[0]

        # add the new tokens to the index
        self.placeholder_view_tokens += [
            placeholder_view_tokens_new[i] for i in idxs_new_tokens
        ]
        self.placeholder_view_token_ids += [
            placeholder_view_token_ids_new[i] for i in idxs_new_tokens
        ]

        # recreate the lookup table WITHOUT rescaling the ranges of the MLP
        self._prepare_view_token_param_lookup(rescale_min_max=False)

    def set_net_style(self,
                      num_unet_layers: int,
                      num_time_anchors: int,
                      output_dim: int = 768):
        # Original-TI (also has arch-code-1)
        if self.original_ti or self.arch_style_net == 1:
            # baseline - TI baseline, which is one thing no matter what.
            assert self.original_ti_init_embed is not None
            if self.output_bypass:
                raise
            self.ti_embeddings = self.original_ti_init_embed.unsqueeze(
                0).repeat(len(self.placeholder_view_token_ids), 1)
            self.ti_embeddings = torch.nn.parameter.Parameter(
                self.ti_embeddings.clone(), requires_grad=True)
            # self.ti_embeddings.register_hook(lambda x: print(x))
            self.lookup_ti_embedding = dict(
                zip(self.placeholder_view_token_ids,
                    torch.arange(len(self.placeholder_view_token_ids))))
            self.output_layer = nn.Identity()  # the MLP aready does projection

        else:
            h_dim = 64
            self.net = nn.Sequential(nn.Linear(self.input_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU(),
                                     nn.Linear(h_dim, h_dim),
                                     nn.LayerNorm(h_dim), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(h_dim, output_dim))
