import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionControlNetPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

import torch
import wandb
from PIL import Image
from tqdm.auto import tqdm

from ..output import StableDiffusionInterpolationOutput
from ..utils import slerp, autogenerate_seed, get_base64_string_from_image_file


class ControlnetInterpolationPipeline(StableDiffusionControlNetPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: (
            ControlNetModel
            | List[ControlNetModel]
            | Tuple[ControlNetModel]
            | MultiControlNetModel
        ),
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
            controlnet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )
        self.seeds = []
        self.verification_responses = []
        self.wandb_table = wandb.Table(
            columns=[
                "idx",
                "generated_frame",
                "verification_response",
                "verification_metadata",
            ]
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

    def verify_generated_frame(
        self, generated_frame: Image.Image, output_dir: str, *args, **kwargs
    ) -> int:
        return True, {}

    def __call__(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]],
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_interpolation_steps: int = 30,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seeds: Optional[List[int]] = None,
        latents: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        output_dir: str = "./outputs",
        max_validation_retries: int = 5,
        **kwargs,
    ):
        os.makedirs(output_dir, exist_ok=True)
        batch_size = len(prompts)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        channels = self.unet.config.in_channels
        controlnet_conditioning_scale = (
            [controlnet_conditioning_scale]
            if isinstance(controlnet_conditioning_scale, float)
            else controlnet_conditioning_scale
        )

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

        interpolated_prompt_embeds, interpolated_negative_prompts_embeds = (
            self.interpolate(
                prompts_embeds=prompts_embeds,
                negative_prompts_embeds=negative_prompts_embeds,
                num_interpolation_steps=num_interpolation_steps,
                batch_size=batch_size,
            )
        )

        self.seeds = seeds or [
            self.autogenerate_seed() for _ in range(len(interpolated_prompt_embeds))
        ]

        # Generating initial U-Net latent vectors from a random normal distribution.
        latents = torch.randn(
            (1, channels, height // 8, width // 8),
            generator=torch.Generator(device="cpu").manual_seed(self.seeds[0]),
        )

        # Generating images using the interpolated embeddings.
        images = []
        for idx, (prompt_embeds, negative_prompt_embeds) in tqdm(
            enumerate(
                zip(interpolated_prompt_embeds, interpolated_negative_prompts_embeds)
            ),
            desc="Generating interpolated frames",
            total=len(interpolated_prompt_embeds),
        ):
            is_regeneration_acceptable = True
            num_retries = 0
            while is_regeneration_acceptable:
                generator = torch.Generator(device="cpu").manual_seed(self.seeds[idx])
                pipeline_output = super().__call__(
                    image=image,
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
                    clip_skip=clip_skip,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    controlnet_conditioning_scale=controlnet_conditioning_scale[idx],
                    guess_mode=guess_mode,
                    control_guidance_start=control_guidance_start,
                    control_guidance_end=control_guidance_end,
                    **kwargs,
                )
                generated_frame = pipeline_output.images[0]
                verification_response, verification_metadata = (
                    self.verify_generated_frame(generated_frame, output_dir)
                )
                self.verification_responses.append(verification_response)
                self.wandb_table.add_data(
                    idx,
                    wandb.Image(generated_frame),
                    verification_response,
                    verification_metadata,
                )
                if verification_response or num_retries >= max_validation_retries:
                    is_regeneration_acceptable = False
                    images.append(generated_frame)
                else:
                    self.seeds[idx] = autogenerate_seed()
                    generator = torch.Generator(device="cpu").manual_seed(
                        self.seeds[idx]
                    )
                    num_retries += 1

        return StableDiffusionInterpolationOutput(frames=images)
