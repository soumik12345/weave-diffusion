import numpy as np
import torch
import wandb
from PIL import Image

from weave_diffusion.stable_diffusion import (
    StableDiffusionImageConditionalInterpolation,
)
from weave_diffusion.utils import autogenerate_seed, log_video


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(
    project="weave-diffusion",
    entity="geekyrakshit",
    job_type="sd/image-cond-interpolation",
)
config = wandb.config
config.num_interpolation_steps = 60
config.height = 1024
config.width = 1024
config.strength = np.linspace(1.0, 0.0, config.num_interpolation_steps).tolist()
config.seed = autogenerate_seed()
generator = torch.manual_seed(config.seed)

pipe = StableDiffusionImageConditionalInterpolation.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
).to(device)
pipe.set_progress_bar_config(leave=False)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Text prompts that describes the desired output image.
prompts = [
    "A cute dog in a beautiful field of lavander colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, kodak portra 400, film grain",
    "A cute cat in a beautiful field of lavander colorful flowers everywhere, perfect lighting, leica summicron 35mm f2.0, kodak portra 400, film grain",
]
# Negative prompts that can be used to steer the generation away from certain features.
negative_prompts = [
    "poorly drawn,cartoon, 2d, sketch, cartoon, drawing, anime, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry",
    "poorly drawn,cartoon, 2d, sketch, cartoon, drawing, anime, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry",
]

conditional_image = Image.open("/workspace/sample_weave_image.png")
conditional_image = conditional_image.resize((config.width, config.height))
frames = pipe(
    prompts=prompts,
    negative_prompts=negative_prompts,
    image=conditional_image,
    strength=config.strength,
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
