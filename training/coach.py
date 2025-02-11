from pathlib import Path
from typing import Optional, Dict, Tuple, List

import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from checkpoint_handler import CheckpointHandler
from constants import UNET_LAYERS
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.neti_mapper import NeTIMapper
from models.xti_attention_processor import XTIAttenProc
from training.config import RunConfig
from training.dataset import TextualInversionDataset
from training.logger import CoachLogger
from training.validate import ValidationHandler
from utils.types import NeTIBatch


class Coach:

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.logger = CoachLogger(cfg=cfg)

        # Initialize some basic stuff
        self.accelerator = self._init_accelerator()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        if self.cfg.optim.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Initialize models
        self.tokenizer, self.noise_scheduler, self.text_encoder, self.vae, self.unet = self._init_sd_models()

        # Initialize dataset and dataloader
        self.train_dataset = self._init_dataset()
        self.train_dataloader = self._init_dataloader(dataset=self.train_dataset)

        # Initialize placeholder tokens
        self.placeholder_style_tokens = self.train_dataset.placeholder_style_tokens
        print(self.placeholder_style_tokens)
        print(len(self.placeholder_style_tokens))
        self.placeholder_object_tokens = self.train_dataset.placeholder_object_tokens
        self.fixed_object_token = self.train_dataset.fixed_object_token
        if self.cfg.eval.validation_style_tokens is not None:
            assert all([
                v in self.placeholder_style_tokens
                for v in self.cfg.eval.validation_style_tokens
            ])

        # add novel concepts
        self.load_pretrained_object_neti = True if self.cfg.data.fixed_object_token_or_path is not None and Path(
            self.cfg.data.fixed_object_token_or_path).suffix == '.pt' else False
        (self.token_embeds, self.placeholder_token_ids, self.placeholder_style_token_ids,
         self.placeholder_object_token_ids) = Coach._add_concept_token_to_tokenizer_static(
            self.cfg, self.train_dataset.placeholder_style_tokens, self.train_dataset.placeholder_object_tokens,
            self.tokenizer, self.text_encoder
        )

        self.cfg.data.placeholder_style_tokens = self.placeholder_style_tokens

        # Initialize mapper and set mapper
        neti_mapper_object_lookup, neti_mapper_style, self.loaded_iteration = self._init_neti_mapper()
        self.text_encoder.text_model.embeddings.set_mapper(neti_mapper_object_lookup, neti_mapper_style)
        self._freeze_all_modules()
        self._set_attn_processor()

        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_scheduler(optimizer=self.optimizer)

        # Prepare everything with accelerator
        self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.text_encoder, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # Reconfigure some parameters that we'll need for training
        self.weight_dtype = self._get_weight_dtype()
        self._set_model_weight_dtypes(weight_dtype=self.weight_dtype)
        self._init_trackers()

        self.validator = ValidationHandler(cfg=self.cfg,
                                           placeholder_style_tokens=self.placeholder_style_tokens,
                                           placeholder_style_token_ids=self.placeholder_style_token_ids,
                                           placeholder_object_tokens=self.placeholder_object_tokens,
                                           placeholder_object_token_ids=self.placeholder_object_token_ids,
                                           fixed_object_token=self.fixed_object_token,
                                           weights_dtype=self.weight_dtype)

        self.checkpoint_handler = CheckpointHandler(cfg=self.cfg,
                                                    placeholder_style_tokens=self.placeholder_style_tokens,
                                                    placeholder_style_token_ids=self.placeholder_style_token_ids,
                                                    placeholder_object_tokens=self.placeholder_object_tokens,
                                                    placeholder_object_token_ids=self.placeholder_object_token_ids,
                                                    save_root=self.cfg.log.exp_dir)

        #### tmp - for testing if the loaded object token works ####
        if 0:
            test_prompts = ["A photo of a <car>"]
            self.validator.infer(accelerator=self.accelerator,
                                 tokenizer=self.tokenizer,
                                 text_encoder=self.text_encoder,
                                 unet=self.unet,
                                 vae=self.vae,
                                 prompts=test_prompts,
                                 num_images_per_prompt=1,
                                 seeds=[42, 420, 501],
                                 step=0)

    def train(self):
        total_batch_size = self.cfg.optim.train_batch_size * self.accelerator.num_processes * \
                           self.cfg.optim.gradient_accumulation_steps
        self.logger.log_start_of_training(total_batch_size=total_batch_size, num_samples=len(self.train_dataset))

        global_step = self._set_global_step()
        progress_bar = tqdm(range(global_step, self.cfg.optim.max_train_steps),
                            disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        orig_embeds_params = self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight.data.clone()
        while global_step < self.cfg.optim.max_train_steps:

            self.text_encoder.train()
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.text_encoder):
                    # Convert images to latent space
                    latent_batch = batch["pixel_values"].to(dtype=self.weight_dtype)
                    latents = self.vae.encode(latent_batch).latent_dist.sample().detach()
                    latents = latents * self.vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(low=0, high=self.noise_scheduler.config.num_train_timesteps,
                                              size=(bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    _hs = self.get_text_conditioning(input_ids=batch['input_ids'],
                                                     input_ids_placeholder_object=batch[
                                                         'input_ids_placeholder_object'],
                                                     input_ids_placeholder_style=batch[
                                                         'input_ids_placeholder_style'],
                                                     timesteps=timesteps,
                                                     device=latents.device)

                    # Predict the noise residual
                    model_pred = self.unet(noisy_latents, timesteps, _hs).sample

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Let's make sure we don't update any embedding weights besides the newly added token
                    # This isn't really needed, but we'll keep it for consistency with the original code
                    index_no_updates = torch.arange(len(self.tokenizer)) != self.placeholder_token_ids
                    with torch.no_grad():
                        self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[
                            index_no_updates] = orig_embeds_params[index_no_updates]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.logger.update_step(step=global_step)
                    if self._should_save(global_step=global_step):
                        self.checkpoint_handler.save_model(text_encoder=self.text_encoder,
                                                           accelerator=self.accelerator,
                                                           embeds_save_name=f"learned_embeds-steps-{global_step}.bin",
                                                           mapper_save_name=f"mapper-steps-{global_step}.pt")

                    if self._should_eval(global_step=global_step):
                        self.validator.infer(accelerator=self.accelerator,
                                             tokenizer=self.tokenizer,
                                             text_encoder=self.text_encoder,
                                             unet=self.unet,
                                             vae=self.vae,
                                             num_images_per_prompt=self.cfg.eval.num_validation_images,
                                             seeds=self.cfg.eval.validation_seeds,
                                             step=global_step)

                logs = {"total_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                if global_step >= self.cfg.optim.max_train_steps:
                    break

        # Save the final model
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.checkpoint_handler.save_model(text_encoder=self.text_encoder,
                                               accelerator=self.accelerator,
                                               embeds_save_name=f"learned_embeds-final.bin",
                                               mapper_save_name=f"mapper-final.pt")
        self.accelerator.end_training()

    def get_text_conditioning(self, input_ids: torch.Tensor,
                              input_ids_placeholder_object: List[int],
                              input_ids_placeholder_style: List[int],
                              timesteps: torch.Tensor,
                              device: torch.device) -> Dict:
        """ Compute the text conditioning for the current batch of images using our text encoder over-ride. """
        _hs = {"this_idx": 0}
        for layer_idx, unet_layer in enumerate(UNET_LAYERS):
            neti_batch = NeTIBatch(
                input_ids=input_ids,
                input_ids_placeholder_object=input_ids_placeholder_object,
                input_ids_placeholder_style=input_ids_placeholder_style,
                timesteps=timesteps,
                unet_layers=torch.tensor(layer_idx, device=device).repeat(timesteps.shape[0])
            )
            layer_hidden_state, layer_hidden_state_bypass = self.text_encoder(batch=neti_batch)

            layer_hidden_state = layer_hidden_state[0].to(dtype=self.weight_dtype)
            _hs[f"CONTEXT_TENSOR_{layer_idx}"] = layer_hidden_state
            if layer_hidden_state_bypass is not None:
                layer_hidden_state_bypass = layer_hidden_state_bypass[0].to(dtype=self.weight_dtype)
                _hs[f"CONTEXT_TENSOR_BYPASS_{layer_idx}"] = layer_hidden_state_bypass
        return _hs

    def _set_global_step(self) -> int:
        global_step = 0
        if self.loaded_iteration is not None:
            global_step = self.loaded_iteration
        self.logger.update_step(step=global_step)
        return global_step

    @staticmethod
    def _add_concept_token_to_tokenizer_static(cfg, placeholder_style_tokens,
                                               placeholder_object_tokens, tokenizer, text_encoder
                                               ):
        """ modifies the tokenizer and text_encoder in place """
        placeholder_tokens = placeholder_style_tokens + placeholder_object_tokens
        num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
        if num_added_tokens == 0:
            raise ValueError(
                f"No new tokens were added to the tokenizer"
                f"Please pass a different `placeholder_token` that is not already in the tokenizer."
            )

        # extract all the placeholder ids
        placeholder_style_token_ids = tokenizer.convert_tokens_to_ids(
            placeholder_style_tokens)

        placeholder_object_token_ids = tokenizer.convert_tokens_to_ids(
            placeholder_object_tokens)
        placeholder_token_ids = tokenizer.convert_tokens_to_ids(
            placeholder_tokens)
        assert set(placeholder_style_token_ids).union(
            set(placeholder_object_token_ids)) == set(placeholder_token_ids)

        # Convert the super_category_token, placeholder_token to ids
        super_category_object_token_id = tokenizer.encode(
            cfg.data.super_category_object_token,
            add_special_tokens=False)
        super_category_style_token_id = tokenizer.encode(
            cfg.data.super_category_style_token, add_special_tokens=False)

        super_token_ids = super_category_object_token_id + super_category_style_token_id

        # Check if super_category_token is a single token or a sequence of tokens
        if len(super_category_object_token_id) != 1:
            raise ValueError(
                f"object supercategory [self.cfg.data.super_category_object_token] not in the vocabulary"
            )
        if len(super_category_style_token_id) != 1:
            raise ValueError(
                "view supercategory [self.cfg.data.super_category_view_token] not in the vocabulary"
            )
        super_category_object_token_id, super_category_style_token_id = super_category_object_token_id[
            0], super_category_style_token_id[0]

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialize the newly added placeholder token with the embeddings of the super category token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        '''
        token_embeds[placeholder_style_token_ids] = token_embeds[
            super_category_style_token_id].clone().unsqueeze(0).repeat(
            len(placeholder_style_token_ids), 1)
        token_embeds[placeholder_object_token_ids] = token_embeds[
            super_category_object_token_id].clone().unsqueeze(0).repeat(
            len(placeholder_object_token_ids), 1)
        '''

        ## Compute the norm of the super category token embedding for scaling mapper output
        cfg.model.target_norm_style = None
        cfg.model.target_norm_object = None

        if cfg.model.normalize_style_mapper_output:
            if super_category_style_token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"super_category_style_token [{cfg.data.super_category_object_token}] ",
                    " is unknown to the tokenizer")
            cfg.model.target_norm_style = token_embeds[
                super_category_style_token_id].norm().item()
        if cfg.model.normalize_object_mapper_output:
            if super_category_object_token_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"super_category_style_token [{super_category_object_token_id}] ",
                    " is unknown to the tokenizer")
            cfg.model.target_norm_object = token_embeds[
                super_category_object_token_id].norm().item()

        return token_embeds, placeholder_token_ids, placeholder_style_token_ids, placeholder_object_token_ids

    def _init_neti_mapper(self) -> Tuple[NeTIMapper, Optional[int]]:
        neti_mapper_object_lookup, neti_mapper_style = None, None
        loaded_iteration = None

        if self.cfg.model.mapper_checkpoint_path:
            # This isn't 100% resuming training since we don't save the optimizer, but it's close enough
            _, neti_mapper = CheckpointHandler.load_mapper(self.cfg.model.mapper_checkpoint_path)
            loaded_iteration = int(self.cfg.model.mapper_checkpoint_path.stem.split("-")[-1])
            print(f"Loaded NeTI mapper checkpoint from iteration: {loaded_iteration}")

        if self.load_pretrained_object_neti:
            _, neti_mapper_object = CheckpointHandler.load_mapper(
                self.cfg.data.fixed_object_token_or_path, "object",
                self.tokenizer)

        # hack
        super_category_style_token = self.cfg.data.super_category_style_token
        original_ti_init_embed = self.text_encoder.get_input_embeddings().weight.data[
            self.tokenizer.convert_tokens_to_ids(super_category_style_token)]

        neti_mapper_style = NeTIMapper(
            embedding_type="style",
            placeholder_style_tokens=self.placeholder_style_tokens,
            placeholder_style_token_ids=self.placeholder_style_token_ids,
            placeholder_object_token=None,
            output_dim=self.cfg.model.word_embedding_dim,
            arch_mlp_hidden_dims=self.cfg.model.arch_mlp_hidden_dims,
            use_nested_dropout=self.cfg.model.use_nested_dropout,
            nested_dropout_prob=self.cfg.model.nested_dropout_prob,
            norm_scale=self.cfg.model.target_norm_style,
            use_positional_encoding=self.cfg.model.
            use_positional_encoding_style,
            num_pe_time_anchors=self.cfg.model.num_pe_time_anchors,
            pe_sigmas=self.cfg.model.pe_sigmas,
            arch_style_net=self.cfg.model.arch_style_net,
            arch_style_mix_streams=self.cfg.model.arch_style_mix_streams,
            arch_style_disable_tl=self.cfg.model.arch_style_disable_tl,
            output_bypass=self.cfg.model.output_bypass_style,
            original_ti=self.cfg.model.original_ti,
            original_ti_init_embed=original_ti_init_embed,
            output_bypass_alpha=self.cfg.model.output_bypass_alpha_style,
            bypass_unconstrained=self.cfg.model.bypass_unconstrained_style)

        return neti_mapper_object_lookup, neti_mapper_style, loaded_iteration

    def _init_sd_models(self):
        tokenizer = self._init_tokenizer()
        noise_scheduler = self._init_noise_scheduler()
        text_encoder = self._init_text_encoder()
        vae = self._init_vae()
        unet = self._init_unet()
        return tokenizer, noise_scheduler, text_encoder, vae, unet

    def _init_tokenizer(self) -> CLIPTokenizer:
        tokenizer = CLIPTokenizer.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        return tokenizer

    def _init_noise_scheduler(self) -> DDPMScheduler:
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="scheduler"
        )
        return noise_scheduler

    def _init_text_encoder(self) -> NeTICLIPTextModel:
        text_encoder = NeTICLIPTextModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="text_encoder", revision=self.cfg.model.revision,
        )
        return text_encoder

    def _init_vae(self) -> AutoencoderKL:
        vae = AutoencoderKL.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=self.cfg.model.revision)
        return vae

    def _init_unet(self) -> UNet2DConditionModel:
        unet = UNet2DConditionModel.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path, subfolder="unet", revision=self.cfg.model.revision
        )
        return unet

    def _freeze_all_modules(self):
        # Freeze vae and unet
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        # Freeze all parameters except for the mapper in text encoder
        self.text_encoder.text_model.encoder.requires_grad_(False)
        self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
        self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        # Make sure to train the mapper
        self.text_encoder.text_model.embeddings.mapper_style.requires_grad_(True)
        self.text_encoder.text_model.embeddings.mapper_style.train()
        if self.cfg.optim.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            self.unet.train()
            self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

    def _set_attn_processor(self):
        self.unet.set_attn_processor(XTIAttenProc())

    def _init_dataset(self) -> TextualInversionDataset:
        dataset = TextualInversionDataset(fixed_object_token_or_path=self.cfg.data.fixed_object_token_or_path,
                                          data_root=self.cfg.data.train_data_dir,
                                          tokenizer=self.tokenizer,
                                          size=self.cfg.data.resolution,
                                          repeats=self.cfg.data.repeats,
                                          center_crop=self.cfg.data.center_crop,
                                          set="train")
        return dataset

    def _init_dataloader(self, dataset: Dataset) -> torch.utils.data.DataLoader:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.cfg.optim.train_batch_size,
                                                 shuffle=True,
                                                 num_workers=self.cfg.data.dataloader_num_workers)
        return dataloader

    def _init_optimizer(self) -> torch.optim.Optimizer:
        if self.cfg.optim.scale_lr:
            self.cfg.optim.learning_rate = (self.cfg.optim.learning_rate *
                                            self.cfg.optim.gradient_accumulation_steps *
                                            self.cfg.optim.train_batch_size *
                                            self.accelerator.num_processes)
        optimizer = torch.optim.AdamW(
            self.text_encoder.text_model.embeddings.mapper_style.parameters(),  # only optimize the embeddings
            lr=self.cfg.optim.learning_rate,
            betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
            weight_decay=self.cfg.optim.adam_weight_decay,
            eps=self.cfg.optim.adam_epsilon,
        )
        return optimizer

    def _init_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        lr_scheduler = get_scheduler(
            self.cfg.optim.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.optim.lr_warmup_steps * self.cfg.optim.gradient_accumulation_steps,
            num_training_steps=self.cfg.optim.max_train_steps * self.cfg.optim.gradient_accumulation_steps,
        )
        return lr_scheduler

    def _init_accelerator(self) -> Accelerator:
        accelerator_project_config = ProjectConfiguration(total_limit=self.cfg.log.checkpoints_total_limit)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.optim.gradient_accumulation_steps,
            mixed_precision=self.cfg.optim.mixed_precision,
            log_with=self.cfg.log.report_to,
            project_config=accelerator_project_config,
        )
        self.logger.log_message(accelerator.state)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        return accelerator

    def _set_model_weight_dtypes(self, weight_dtype: torch.dtype):
        self.unet.to(self.accelerator.device, dtype=weight_dtype)
        self.vae.to(self.accelerator.device, dtype=weight_dtype)

    def _get_weight_dtype(self) -> torch.dtype:
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype

    def _init_trackers(self):
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("textual_inversion")

    def _should_save(self, global_step: int) -> bool:
        return global_step % self.cfg.log.save_steps == 0

    def _should_eval(self, global_step: int) -> bool:
        return self.cfg.eval.validation_prompts is not None and global_step % self.cfg.eval.validation_steps == 0
