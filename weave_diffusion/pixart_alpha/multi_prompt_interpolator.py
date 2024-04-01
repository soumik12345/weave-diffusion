from typing import Callable, List, Optional, Union

from diffusers import PixArtAlphaPipeline
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from transformers import T5EncoderModel, T5Tokenizer

import torch
from tqdm.auto import tqdm

from ..output import PixartAlphaInterpolationOutput
from ..utils import slerp


class PixArtAlphaMultiPromptInterpolationPipeline(PixArtAlphaPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

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
        num_interpolation_steps: int = 30,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 120,
        **kwargs
    ) -> PixartAlphaInterpolationOutput:
        batch_size = len(prompts)
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        channels = self.transformer.config.in_channels

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

        latents = torch.randn(
            (
                1,
                channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ),
            generator=generator,
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
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                output_type="pil",
                return_dict=return_dict,
                callback=callback,
                callback_steps=callback_steps,
                clean_caption=clean_caption,
                use_resolution_binning=use_resolution_binning,
                max_sequence_length=max_sequence_length,
                **kwargs
            )
            generated_frame = pipeline_output.images[0]
            images.append(generated_frame)

        return PixartAlphaInterpolationOutput(frames=images)
