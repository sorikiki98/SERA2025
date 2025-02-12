from time import sleep

import ipdb
from typing import List

import numpy as np
from requests.exceptions import ConnectionError
import torch
from PIL import Image
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import is_wandb_available
from tqdm import tqdm
from transformers import CLIPTokenizer

import matplotlib.pyplot as plt
from training.config import RunConfig
from models.neti_clip_text_encoder import NeTICLIPTextModel
from models.xti_attention_processor import XTIAttenProc
from prompt_manager import PromptManager
from sd_pipeline_call import sd_pipeline_call
from torchvision.utils import make_grid

if is_wandb_available():
    import wandb


class ValidationHandler:

    def __init__(self,
                 cfg: RunConfig,
                 placeholder_style_tokens: List[str],
                 placeholder_style_token_ids: List[int],
                 placeholder_object_tokens: List[str],
                 placeholder_object_token_ids: List[int],
                 fixed_object_token: str,
                 weights_dtype: torch.dtype,
                 max_rows: int = 14,
                 ):

        self.cfg = cfg
        self.placeholder_style_tokens = placeholder_style_tokens
        self.placeholder_style_token_ids = placeholder_style_token_ids
        self.placeholder_object_tokens = placeholder_object_tokens
        self.placeholder_object_token_ids = placeholder_object_token_ids
        self.fixed_object_token = fixed_object_token
        self.weight_dtype = weights_dtype
        self.max_rows = max_rows
        self.size = 512

    def infer(self,
              accelerator: Accelerator,
              tokenizer: CLIPTokenizer,
              text_encoder: NeTICLIPTextModel,
              unet: UNet2DConditionModel,
              vae: AutoencoderKL,
              num_images_per_prompt: int,
              seeds: List[int],
              step: int,
              prompts: List[str] = None):
        """ Runs inference during our training scheme. """
        try:
            pipeline = self.load_stable_diffusion_model(
                accelerator, tokenizer, text_encoder, unet, vae)
        except ConnectionError as e:
            try:
                sleep(60 * 5)
                pipeline = self.load_stable_diffusion_model(
                    accelerator, tokenizer, text_encoder, unet, vae)
            except ConnectionError as e:
                print("Connection error, resuming")
                print(e)
                return None

        prompt_manager = PromptManager(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            timesteps=pipeline.scheduler.timesteps,
            placeholder_style_token_ids=self.placeholder_style_token_ids,
            placeholder_object_token_ids=self.placeholder_object_token_ids,
        )

        if prompts is None:
            if self.cfg.eval.validation_style_tokens is not None:
                style_tokens = self.cfg.eval.validation_style_tokens
            else:
                style_tokens = self.placeholder_style_tokens

            if len(style_tokens) > 100:
                style_tokens = style_tokens[::30].copy()

            style_tokens = style_tokens[:self.max_rows - 1].copy()

            prompts = [
                f"A photo of a {v} {self.fixed_object_token}"
                for v in style_tokens
            ]
        else:
            formatted_prompts = []
            for prompt in prompts:
                formatted_prompt = prompt.format(self.fixed_object_token)
                formatted_prompts.append(formatted_prompt)
            prompts = formatted_prompts

        print(prompts)
        joined_images = []

        for prompt in prompts:
            images = self.infer_on_prompt(
                pipeline=pipeline,
                prompt_manager=prompt_manager,
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                seeds=seeds)
            prompt_image = Image.fromarray(np.concatenate(images, axis=1))
            joined_images.append(prompt_image)

        final_image = Image.fromarray(np.concatenate(joined_images, axis=0))
        final_image.save(self.cfg.log.exp_dir / f"val-image-{step}.png")
        self.log_with_accelerator(accelerator, joined_images, step=step, prompts=prompts)
        del pipeline
        torch.cuda.empty_cache()
        if text_encoder.text_model.embeddings.mapper_style is not None:
            text_encoder.text_model.embeddings.mapper_style.train()
        # change dict to ModuleDIct
        if text_encoder.text_model.embeddings.mapper_object_lookup is not None:
            for mapper in text_encoder.text_model.embeddings.mapper_object_lookup.values():
                mapper.train()

            # for mapper in (text_encoder.text_model.embeddings.mapper_object,
            #                text_encoder.text_model.embeddings.mapper_style):
            if mapper is not None:
                mapper.train()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        return final_image

    def infer_on_prompt(self,
                        pipeline: StableDiffusionPipeline,
                        prompt_manager: PromptManager,
                        prompt: str,
                        seeds: List[int],
                        num_images_per_prompt: int = 1) -> List[Image.Image]:
        prompt_embeds = self.compute_embeddings(prompt_manager=prompt_manager,
                                                prompt=prompt)
        all_images = []
        for idx in tqdm(range(num_images_per_prompt)):
            generator = torch.Generator(device='cuda').manual_seed(seeds[idx])
            images = sd_pipeline_call(
                pipeline,
                prompt_embeds=prompt_embeds,
                generator=generator,
                num_inference_steps=self.cfg.eval.num_denoising_steps,
                num_images_per_prompt=1,
                height=self.size,
                width=self.size).images
            all_images.extend(images)
        return all_images


    @staticmethod
    def compute_embeddings(prompt_manager: PromptManager,
                           prompt: str) -> torch.Tensor:
        with torch.autocast("cuda"):
            with torch.no_grad():
                prompt_embeds = prompt_manager.embed_prompt(prompt)
        return prompt_embeds

    def load_stable_diffusion_model(
            self, accelerator: Accelerator, tokenizer: CLIPTokenizer,
            text_encoder: NeTICLIPTextModel, unet: UNet2DConditionModel,
            vae: AutoencoderKL) -> StableDiffusionPipeline:
        """ Loads SD model given the current text encoder and our mapper. """
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model.pretrained_model_name_or_path,
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            torch_dtype=self.weight_dtype)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config)
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler.set_timesteps(self.cfg.eval.num_denoising_steps,
                                         device=pipeline.device)
        pipeline.unet.set_attn_processor(XTIAttenProc())
        if text_encoder.text_model.embeddings.mapper_style is not None:
            text_encoder.text_model.embeddings.mapper_style.eval()
        # change dict to ModuleDIct
        if text_encoder.text_model.embeddings.mapper_object_lookup is not None:
            for mapper in text_encoder.text_model.embeddings.mapper_object_lookup.values():
                mapper.eval()
        return pipeline

    def log_with_accelerator(self, accelerator: Accelerator,
                             images: List[Image.Image], step: int,
                             prompts: List[str]):
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation",
                                          np_images,
                                          step,
                                          dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log({
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                })
