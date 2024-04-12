import os
from typing import List, Tuple

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from weave_diffusion.stable_diffusion import ControlnetInterpolationPipeline
from weave_diffusion.utils import (
    autogenerate_seed,
    center_crop,
    get_base64_string_from_image_file,
    log_video,
)

import cv2
import fire
import numpy as np
import torch
import wandb
from PIL import Image


class VLMAssistedControlnetInterpolationPipeline(ControlnetInterpolationPipeline):

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
        self.multi_modal_model = ChatAnthropic(model="claude-3-opus-20240229")
        self.multi_modal_system_message = SystemMessage(
            content="""You are an expert at determining anomalies in AI-generated images.
            Your goal is to carefully observe the image and describe any answers asked in the context of AI-generated images to the point."""
        )

    def verify_generated_frame(
        self, generated_frame: Image, output_dir: str, *args, **kwargs
    ) -> int:
        generated_frame_path = os.path.join(output_dir, f"frame.png")
        generated_frame.save(generated_frame_path)
        encoded_frame = get_base64_string_from_image_file(generated_frame_path)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "How many circles do you see in the image? Answer with just the exact number.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_frame}"},
                },
            ]
        )
        response = int(
            self.multi_modal_model.invoke(
                [self.multi_modal_system_message, message]
            ).content
        )
        return response <= 12, {"circles_count": response}


def generate(
    prompts: List[str] = [
        "threads of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "threads of cotton on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "threads of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "balls of cotton on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "balls of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "yellow cotton balls on a black matte flat graphic design background, realistic rendering, 4K, imax, unreal engine, sharp, bright and contrasting colors",
        "yellow woolen balls on a black matte flat graphic design background, realistic rendering, 4K, imax, logo design, sharp, bright and contrasting colors",
    ],
    negative_prompts: List[str] = [
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, blue, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    ],
    image: str = "logo.png",
    canny_image: str = "logo_white.png",
    controlnet_model_address: str = "lllyasviel/sd-controlnet-canny",
    generation_model_address: str = "runwayml/stable-diffusion-v1-5",
    apply_vlm_assistance: bool = False,
    num_interpolation_steps: int = 5,
    height: int = 1024,
    width: int = 1024,
    canny_low_threshold: int = 100,
    canny_high_threshold: int = 200,
    wandb_project: str = "weave-diffusion",
    wandb_job_type: str = "controlnet",
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    wandb.init(project=wandb_project, job_type=wandb_job_type)
    config = wandb.config
    config.num_interpolation_steps = num_interpolation_steps
    config.height = height
    config.width = width
    config.canny_low_threshold = canny_low_threshold
    config.canny_high_threshold = canny_high_threshold

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_address,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe = (
        VLMAssistedControlnetInterpolationPipeline.from_pretrained(
            generation_model_address,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
            use_safetensors=True,
        ).to(device)
        if apply_vlm_assistance
        else ControlnetInterpolationPipeline.from_pretrained(
            generation_model_address,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
            use_safetensors=True,
        ).to(device)
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(leave=False)
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    image = np.array(Image.open(image))[:, :, :3]
    image = center_crop(image, 50)
    image = cv2.resize(
        image, (config.height, config.width), interpolation=cv2.INTER_LINEAR
    )
    canny_image = Image.open(canny_image)

    config.prompts = prompts
    config.negative_prompts = negative_prompts

    config.num_frames = (len(config.prompts) - 1) * config.num_interpolation_steps
    # config.controlnet_conditioning_scale = np.linspace(
    #     0.0, 1.0, config.num_frames // 2
    # ).tolist() + [1.0] * (config.num_frames // 2)
    config.controlnet_conditioning_scale = [1.0] * config.num_frames

    api = wandb.Api()
    run = api.run("geekyrakshit/weave-diffusion/l035qocc")
    config_dict = run.config

    frames = pipe(
        prompts=config.prompts,
        negative_prompts=config.negative_prompts,
        image=canny_image,
        num_interpolation_steps=config.num_interpolation_steps,
        seeds=[autogenerate_seed()] * config.num_frames,
        controlnet_conditioning_scale=config.controlnet_conditioning_scale,
    ).frames

    config.seeds = pipe.seeds

    video_path = log_video(images=frames, save_path="./output")
    video = wandb.Video(video_path)
    table = wandb.Table(
        columns=[
            "input_image",
            "canny_image",
            "prompts",
            "negative_prompts",
            "interpolated_video",
        ]
    )
    table.add_data(
        wandb.Image(image),
        wandb.Image(canny_image),
        config.prompts,
        config.negative_prompts,
        video,
    )
    loggable_dict = {"Interpolated-Video": video, "Result-Table": table}
    if isinstance(pipe, VLMAssistedControlnetInterpolationPipeline):
        loggable_dict["Debug_table"] = pipe.wandb_table
    wandb.log(loggable_dict)

    artifact = wandb.Artifact(name=f"video-{wandb.run.id}", type="video")
    artifact.add_file(local_path=video_path)
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    fire.Fire(generate)
