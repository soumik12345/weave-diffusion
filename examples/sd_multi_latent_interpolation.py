import torch
import wandb

from weave_diffusion.stable_diffusion import (
    StableDiffusionMultiLatentInterpolationPipeline,
)
from weave_diffusion.utils import autogenerate_seed, log_video


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(
    project="weave-diffusion",
    entity="geekyrakshit",
    job_type="sd/multi-latent-interpolation",
)
config = wandb.config
config.num_interpolation_steps = 60
config.height = 1024
config.width = 1024
config.seed = autogenerate_seed()
generator = torch.manual_seed(config.seed)

pipe = StableDiffusionMultiLatentInterpolationPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
).to(device)
pipe.set_progress_bar_config(leave=False)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "Sci-fi digital painting of an alien landscape with otherworldly plants, strange creatures, and distant planets."
negative_prompt = "poorly drawn,cartoon, 3d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"

frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_interpolation_steps=config.num_interpolation_steps,
).frames

video_path = log_video(images=frames, save_path="./output")
video = wandb.Video(video_path)
table = wandb.Table(columns=["prompts", "negative_prompts", "interpolated_video"])
table.add_data(prompt, negative_prompt, video)
wandb.log({"Interpolated-Video": video, "Result-Table": table})

artifact = wandb.Artifact(name=f"video-{wandb.run.id}", type="video")
artifact.add_dir(local_path=video_path)
wandb.log_artifact(artifact)
