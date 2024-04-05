import torch
import wandb

from weave_diffusion.stable_diffusion import (
    StableDiffusionMultiPromptInterpolationPipeline,
)
from weave_diffusion.utils import autogenerate_seed, log_video


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(
    project="weave-diffusion",
    entity="geekyrakshit",
    job_type="sd/multi-prompt-interpolation",
)
config = wandb.config
config.num_interpolation_steps = 60
config.height = 1920
config.width = 1080
config.seed = autogenerate_seed()
generator = torch.manual_seed(config.seed)

pipe = StableDiffusionMultiPromptInterpolationPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
).to(device)
pipe.set_progress_bar_config(leave=False)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Text prompts that describes the desired output image.
prompts = [
    "a network of woolen balls connected by threads looking like cells, realistic rendering, color palette consisting of apricot, yello, cyan, pink, and white",
    "several woven woolen blankets of colors apricot, yello, cyan, pink, and white lying in a stack",
]
# Negative prompts that can be used to steer the generation away from certain features.
negative_prompts = [
    "metal, squirming, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    "metal, squirming, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
]

frames = pipe(
    prompts=prompts,
    negative_prompts=negative_prompts,
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
