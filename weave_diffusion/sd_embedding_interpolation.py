from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from tqdm.auto import tqdm

from .output import StableDiffusionInterpolationOutput


class StableDiffusionLatentWalkerPipeline(StableDiffusionPipeline):

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

    def __call__(
        self,
        prompt: str = None,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        interpolation_step_size: float = 0.001,
        num_interpolation_steps: int = 30,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
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
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        channels = self.unet.config.in_channels

        # Tokenizing and encoding the prompt into embeddings.
        prompt_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(prompt_tokens.input_ids.to(self.device))[0]

        # Tokenizing and encoding the negative prompt into embeddings.
        if negative_prompt is None:
            negative_prompt = [""]

        negative_prompt_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        negative_prompt_embeds = self.text_encoder(
            negative_prompt_tokens.input_ids.to(self.device)
        )[0]

        # Generating initial latent vectors from a random normal distribution,
        # with the option to use a generator for reproducibility.
        latents = torch.randn(
            (1, channels, height // 8, width // 8), generator=generator
        )

        walked_embeddings = []

        # Interpolating between embeddings for the given number of interpolation steps.
        for i in range(num_interpolation_steps):
            walked_embeddings.append(
                [
                    prompt_embeds + interpolation_step_size * i,
                    negative_prompt_embeds + interpolation_step_size * i,
                ]
            )

        # Generating images using the interpolated embeddings.
        images = []
        for latent in tqdm(
            walked_embeddings,
            desc="Generating interpolated frames",
            total=num_interpolation_steps,
        ):
            pipeline_output = super().__call__(
                height=height,
                width=width,
                num_images_per_prompt=1,
                prompt_embeds=latent[0],
                negative_prompt_embeds=latent[1],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                latents=latents,
                timesteps=timesteps,
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
