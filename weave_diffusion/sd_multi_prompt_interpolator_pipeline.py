from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.outputs import BaseOutput

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

import torch
from tqdm.auto import tqdm
from PIL import Image

from .utils import slerp


@dataclass
class StableDiffusionInterpolationOutput(BaseOutput):
    frames: List[Image.Image]


class StableDiffusionMultiPromptInterpolationPipeline(StableDiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )

    def interpolate(
        self,
        prompts_embeds,
        negative_prompts_embeds,
        num_interpolation_steps,
        batch_size,
    ):
        # Interpolating between embeddings pairs for the given number of interpolation steps.
        interpolated_prompt_embeds = []
        interpolated_negative_prompts_embeds = []
        for i in range(batch_size - 1):
            interpolated_prompt_embeds.append(
                slerp(prompts_embeds[i], prompts_embeds[i + 1], num_interpolation_steps)
            )
            interpolated_negative_prompts_embeds.append(
                slerp(
                    negative_prompts_embeds[i],
                    negative_prompts_embeds[i + 1],
                    num_interpolation_steps,
                )
            )
        interpolated_prompt_embeds = torch.cat(interpolated_prompt_embeds, dim=0).to(
            self.device
        )
        interpolated_negative_prompts_embeds = torch.cat(
            interpolated_negative_prompts_embeds, dim=0
        ).to(self.device)
        return interpolated_prompt_embeds, interpolated_negative_prompts_embeds

    def __call__(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_interpolation_steps: int = 30,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ) -> StableDiffusionInterpolationOutput:
        batch_size = len(prompts)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        channels = self.unet.config.in_channels

        # Tokenizing and encoding prompts into embeddings.
        prompts_tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompts_embeds = self.text_encoder(prompts_tokens.input_ids.to(self.device))[0]

        # Tokenizing and encoding negative prompts into embeddings.
        if negative_prompts is None:
            negative_prompts = [""] * batch_size

        negative_prompts_tokens = self.tokenizer(
            negative_prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompts_embeds = self.text_encoder(
            negative_prompts_tokens.input_ids.to(self.device)
        )[0]

        # Generating initial U-Net latent vectors from a random normal distribution.
        latents = torch.randn(
            (1, channels, height // 8, width // 8), generator=generator
        )

        interpolated_prompt_embeds, interpolated_negative_prompts_embeds = (
            self.interpolate(
                prompts_embeds=prompts_embeds,
                negative_prompts_embeds=negative_prompts_embeds,
                num_interpolation_steps=num_interpolation_steps,
                batch_size=batch_size,
            )
        )

        # Generating images using the interpolated embeddings.
        images = []
        for prompt_embeds, negative_prompt_embeds in tqdm(
            zip(interpolated_prompt_embeds, interpolated_negative_prompts_embeds),
            desc="Generating interpolated frames",
            total=len(interpolated_prompt_embeds),
        ):
            pipeline_output = super().__call__(
                height=height,
                width=width,
                num_images_per_prompt=1,
                prompt_embeds=prompt_embeds[None, ...],
                negative_prompt_embeds=negative_prompt_embeds[None, ...],
                generator=generator,
                latents=latents,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                eta=eta,
                ip_adapter_image=ip_adapter_image,
                ip_adapter_image_embeds=ip_adapter_image_embeds,
                output_type="pil",
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                **kwargs,
            )
            generated_frame = pipeline_output.images[0]
            images.append(generated_frame)

        return StableDiffusionInterpolationOutput(frames=images)
