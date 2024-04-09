import cv2
import numpy as np
import torch
import wandb
from PIL import Image

from diffusers import ControlNetModel, UniPCMultistepScheduler

from weave_diffusion.stable_diffusion import ControlnetInterpolationPipeline
from weave_diffusion.utils import (
    autogenerate_seed,
    center_crop,
    convert_to_canny,
    log_video,
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(
    project="weave-diffusion",
    entity="geekyrakshit",
    job_type="controlnet",
)
config = wandb.config
config.num_interpolation_steps = 45
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
    use_safetensors=True,
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(leave=False)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

image = np.array(Image.open("logo.png"))[:, :, :3]
image = center_crop(image, 50)
image = cv2.resize(image, (config.height, config.width), interpolation=cv2.INTER_LINEAR)
canny_image = convert_to_canny(
    image, config.canny_low_threshold, config.canny_high_threshold
)

config.prompts = [
    "balls of twine, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "balls of wool, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "balls of twine lying on a gloden and black blanket, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "balls of wool with needles lying on a gloden and black blanket, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
    "stars in space, 4K, imax, sharp, bright and contrasting colors, sharp, bright and contrasting colors",
    "yellow twine balls on a white background, realistic rendering, 4K, imax, unreal engine, sharp, bright and contrasting colors",
    "yellow woolen balls on a white background, realistic rendering, 4K, imax, logo design, sharp, bright and contrasting colors",
]
# Negative prompts that can be used to steer the generation away from certain features.
config.negative_prompts = [
    "apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "blue, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "blue, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "black hole, nebula, clouds, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
]

config.seeds = [
    autogenerate_seed()
    for _ in range((len(config.prompts) - 1) * config.num_interpolation_steps)
]

frames = pipe(
    prompts=config.prompts,
    negative_prompts=config.negative_prompts,
    image=canny_image,
    num_interpolation_steps=config.num_interpolation_steps,
    seeds=config.seeds,
).frames

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
wandb.log({"Interpolated-Video": video, "Result-Table": table})

artifact = wandb.Artifact(name=f"video-{wandb.run.id}", type="video")
artifact.add_file(local_path=video_path)
wandb.log_artifact(artifact)
