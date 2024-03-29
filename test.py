import torch
import wandb

from weave_diffusion import StableDiffusionPromptInterpolationPipeline
from weave_diffusion.utils import autogenerate_seed, log_video


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(project="weave-diffusion", entity="geekyrakshit", job_type="sd_slerp")
config = wandb.config
config.num_interpolation_steps = 30
config.height = 512
config.width = 512
config.seed = 825375911  # autogenerate_seed()
generator = torch.manual_seed(config.seed)

pipe = StableDiffusionPromptInterpolationPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
).to(device)
pipe.set_progress_bar_config(disable=True)
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

frames = pipe(
    prompts=prompts,
    negative_prompts=negative_prompts,
    num_interpolation_steps=config.num_interpolation_steps,
).frames

video = wandb.Video(log_video(images=frames, save_path="./output"))
table = wandb.Table(columns=["prompts", "negative_prompts", "interpolated_video"])
table.add_data(prompts, negative_prompts, video)
wandb.log({"Interpolated-Video": video})
