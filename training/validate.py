from time import sleep

import PIL
import ipdb
from typing import List, Dict, Any

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
from transformers import CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

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
                 placeholder_new_tokens: List[str],
                 placeholder_new_token_ids: List[int],
                 weights_dtype: torch.dtype,
                 device: torch.device,
                 max_rows: int = 14,
                 ):

        self.cfg = cfg
        self.placeholder_img_tokens = placeholder_new_tokens
        self.placeholder_img_token_ids = placeholder_new_token_ids
        self.weight_dtype = weights_dtype
        self.device = device
        self.max_rows = max_rows
        self.size = 512

    def infer(self,
              tokenizer: CLIPTokenizer,
              text_encoder: NeTICLIPTextModel,
              unet: UNet2DConditionModel,
              vae: AutoencoderKL,
              image_encoder: CLIPVisionModelWithProjection,
              image_processor: CLIPImageProcessor,
              num_images_per_prompt: int,
              seeds: List[int],
              step: int,
              prompts: List[str] = None,
              image_paths: List[str] = None):
        """ Runs inference during our training scheme. """
        try:
            pipeline = self.load_stable_diffusion_model(
                tokenizer, text_encoder, unet, vae)
        except ConnectionError as e:
            try:
                sleep(60 * 5)
                pipeline = self.load_stable_diffusion_model(
                    tokenizer, text_encoder, unet, vae)
            except ConnectionError as e:
                print("Connection error, resuming")
                print(e)
                return None

        prompt_manager = PromptManager(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            timesteps=pipeline.scheduler.timesteps,
            placeholder_img_token_ids=self.placeholder_img_token_ids
        )

        joined_images = []

        for prompt in prompts:
            for path in image_paths:
                with open(path, 'rb') as f:
                    img = Image.open(f).convert('RGB')
                    processed_img = image_processor(img)
                    img_pixel_values = torch.from_numpy(processed_img['pixel_values'][0])

                img_pixel_values = img_pixel_values.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    img_features = image_encoder(img_pixel_values.to(image_encoder.dtype))

                image_embeds = img_features.image_embeds.float()

                images = self.infer_on_prompt(
                    pipeline=pipeline,
                    prompt_manager=prompt_manager,
                    prompt=prompt,
                    image_embeds=image_embeds,
                    num_images_per_prompt=num_images_per_prompt,
                    seeds=seeds)
                prompt_image = Image.fromarray(np.concatenate(images, axis=1))
                joined_images.append(prompt_image)

        final_image = Image.fromarray(np.concatenate(joined_images, axis=0))
        final_image.save(self.cfg.log.exp_dir / f"val-image-{step}.png")
        # self.log_with_accelerator(accelerator, joined_images, step=step, prompts=prompts)
        del pipeline
        torch.cuda.empty_cache()
        if text_encoder.text_model.embeddings.mapper is not None:
            text_encoder.text_model.embeddings.mapper.train()
        if self.cfg.optim.seed is not None:
            set_seed(self.cfg.optim.seed)
        return final_image

    def infer_on_prompt(self,
                        pipeline: StableDiffusionPipeline,
                        prompt_manager: PromptManager,
                        prompt: str,
                        image_embeds: torch.Tensor,
                        seeds: List[int],
                        num_images_per_prompt: int = 1) -> List[Image.Image]:

        prompt_embeds = self.compute_embeddings(prompt_manager=prompt_manager,
                                                prompt=prompt,
                                                image_embeds=image_embeds)
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
                           prompt: str,
                           image_embeds: torch.Tensor,
                           ) -> List[Dict[str, Any]]:
        with torch.autocast("cuda"):
            with torch.no_grad():
                prompt_embeds = prompt_manager.embed_prompt(image_embeds, prompt)
        return prompt_embeds

    def load_stable_diffusion_model(
            self, tokenizer: CLIPTokenizer,
            text_encoder: NeTICLIPTextModel, unet: UNet2DConditionModel,
            vae: AutoencoderKL) -> StableDiffusionPipeline:
        """ Loads SD model given the current text encoder and our mapper. """
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.cfg.model.pretrained_diffusion_model_name_or_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            vae=vae,
            torch_dtype=self.weight_dtype, safety_checker=None
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config)
        pipeline = pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler.set_timesteps(self.cfg.eval.num_denoising_steps,
                                         device=pipeline.device)
        pipeline.unet.set_attn_processor(XTIAttenProc())
        if text_encoder.text_model.embeddings.mapper is not None:
            text_encoder.text_model.embeddings.mapper.eval()
        return pipeline
