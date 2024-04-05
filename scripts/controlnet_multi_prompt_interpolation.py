import cv2
import numpy as np
import torch
import wandb
from PIL import Image

from diffusers.utils import load_image
from diffusers import ControlNetModel, UniPCMultistepScheduler

from weave_diffusion.stable_diffusion import ControlnetInterpolationPipeline
from weave_diffusion.utils import autogenerate_seed, log_video


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(
    project="weave-diffusion",
    entity="geekyrakshit",
    job_type="controlnet",
)
config = wandb.config
config.num_interpolation_steps = 30
config.height = 512
config.width = 512
config.seed = autogenerate_seed()
config.canny_low_threshold = 100
config.canny_high_threshold = 200
generator = torch.manual_seed(config.seed)

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
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(leave=False)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

original_image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)

image = np.array(original_image)

image = cv2.Canny(image, config.canny_low_threshold, config.canny_high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

prompts = [
    "priyanka chopra, best quality, extremely detailed",
    "taylor swift, best quality, extremely detailed",
]
# Negative prompts that can be used to steer the generation away from certain features.
negative_prompts = [
    "monochrome, lowres, bad anatomy, worst quality, low quality",
    "monochrome, lowres, bad anatomy, worst quality, low quality",
]

frames = pipe(
    prompts=prompts,
    negative_prompts=negative_prompts,
    image=canny_image,
    num_interpolation_steps=config.num_interpolation_steps,
).frames

video_path = log_video(images=frames, save_path="./output")
video = wandb.Video(video_path)
table = wandb.Table(columns=["prompts", "negative_prompts", "interpolated_video"])
table.add_data(prompts, negative_prompts, video)
wandb.log({"Interpolated-Video": video, "Result-Table": table})

artifact = wandb.Artifact(name=f"video-{wandb.run.id}", type="video")
artifact.add_file(local_path=video_path)
wandb.log_artifact(artifact)
