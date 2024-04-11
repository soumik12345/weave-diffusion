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
            Your goal is to carefully observe the image and describe the exact number of circles in the generated frames."""
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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(
    project="weave-diffusion",
    entity="geekyrakshit",
    job_type="controlnet",
)
config = wandb.config
config.num_interpolation_steps = 5
config.height = 1024
config.width = 1024
config.canny_low_threshold = 100
config.canny_high_threshold = 200

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32,
    use_safetensors=True,
)
pipe = ControlnetInterpolationPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None,
    use_safetensors=True,
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(leave=False)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

image = np.array(Image.open("logo.png"))[:, :, :3]
image = center_crop(image, 50)
image = cv2.resize(image, (config.height, config.width), interpolation=cv2.INTER_LINEAR)
canny_image = Image.open("logo_white.png")


config.prompts = [
    "threads of woolen on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "threads of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "threads of woolen and wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "balls of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "balls of woolen on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "yellow wool balls on a black matte flat graphic design background, realistic rendering, 4K, imax, unreal engine, sharp, bright and contrasting colors",
    "yellow woolen balls on a black matte flat graphic design background, realistic rendering, 4K, imax, logo design, sharp, bright and contrasting colors",
]
# Negative prompts that can be used to steer the generation away from certain features.
config.negative_prompts = [
    "text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "text, blue, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "text, metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "text, metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
]

config.num_frames = (len(config.prompts) - 1) * config.num_interpolation_steps
# config.controlnet_conditioning_scale = np.linspace(
#     0.0, 1.0, config.num_frames // 2
# ).tolist() + [1.0] * (config.num_frames // 2)
config.controlnet_conditioning_scale = [1.0] * config.num_frames

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
wandb.log(
    {
        "Interpolated-Video": video,
        "Result-Table": table,
        "Debug_table": pipe.wandb_table,
    }
)

artifact = wandb.Artifact(name=f"video-{wandb.run.id}", type="video")
artifact.add_file(local_path=video_path)
wandb.log_artifact(artifact)
