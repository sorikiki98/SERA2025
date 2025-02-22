import ipdb
from typing import Optional, Tuple, Dict, Union, Any

import torch
from torch import nn, Tensor
from transformers import CLIPTextConfig

from models.neti_mapper import NeTIMapper
from utils.types import NeTIBatch, MapperOutput


class NeTICLIPTextEmbeddings(nn.Module):
    """ Modification of CLIPTextEmbedding to allow for the use of a NeTIMapper to overwrite the concept token. """

    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings,
                                               embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)))

    def set_mapper(self, mapper: NeTIMapper, device='cuda'):
        self.mapper = mapper

        # because its a dict, it won't put the mappers on cuda automatically
        # for k, v in self.mapper_object_lookup.items():
        #     self.mapper_object_lookup[k] = v.to(device)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            batch: Optional[NeTIBatch] = None
    ) -> Tuple[Union[Tensor, Any], Optional[Any], Union[bool, Any], Optional[Any]]:

        if batch is not None:
            input_ids = batch.input_ids

        seq_length = input_ids.shape[
            -1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)  # [2, 77, 768]

        ####################################################################
        # NeTI logic - Use mapper to overwrite the learnable token embedding
        ####################################################################
        bypass_outputs = None
        bypass_unconstrained = False
        output_bypass_alpha = None

        if batch is not None:
            if self.mapper is not None:
                assert torch.all(batch.input_ids_placeholder_img ==
                                 batch.input_ids_placeholder_img[0])

                # compute the (t,l)-conditioned embedding
                mapper = self.mapper(
                    timestep=batch.timesteps.float(),
                    unet_layer=batch.unet_layers.float(),
                    image_embeds=batch.image_embeds,
                    truncation_idx=batch.truncation_idx,
                )  # [bs, 1536]
                # strength of the output bypass -> to pass up to the encoder
                output_bypass_alpha = mapper.output_bypass_alpha

                # flag for whether we have the 'bypass_unconstrained' training mode
                bypass_unconstrained = mapper.bypass_unconstrained

                # word embedding vector
                word_embedding = mapper.word_embedding.to(
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # output_bypass if that flag is on
                if mapper.output_bypass:
                    bypass_outputs = mapper.bypass_output.to(
                        dtype=inputs_embeds.dtype, device=inputs_embeds.device)

                # replace special token embedding.
                locs = (input_ids ==
                        batch.input_ids_placeholder_img.unsqueeze(1))
                # assert all(locs.sum(1) == 1)
                if all(locs.sum(1) == 1):
                    inputs_embeds[torch.where(locs)] = word_embedding

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings, bypass_outputs, bypass_unconstrained, output_bypass_alpha
