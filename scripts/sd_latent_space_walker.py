import torch
import wandb

from weave_diffusion.stable_diffusion import StableDiffusionLatentWalkerPipeline
from weave_diffusion.utils import autogenerate_seed, log_video


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

wandb.init(
    project="weave-diffusion", entity="geekyrakshit", job_type="sd/latent-walker"
)
config = wandb.config
config.interpolation_step_size = 0.001
config.num_interpolation_steps = 60
config.height = 1024
config.width = 1024
config.seed = autogenerate_seed()
generator = torch.manual_seed(config.seed)

pipe = StableDiffusionLatentWalkerPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
).to(device)
pipe.set_progress_bar_config(leave=False)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "Epic shot of Sweden, ultra detailed lake with an ren dear, nostalgic vintage, ultra cozy and inviting, wonderful light atmosphere, fairy, little photorealistic, digital painting, sharp focus, ultra cozy and inviting, wish to be there. very detailed, arty, should rank high on youtube for a dream trip."
negative_prompt = "poorly drawn,cartoon, 2d, disfigured, bad art, deformed, poorly drawn, extra limbs, close up, b&w, weird colors, blurry"

frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_interpolation_steps=config.num_interpolation_steps,
    interpolation_step_size=config.interpolation_step_size,
).frames

video_path = log_video(images=frames, save_path="./output")
video = wandb.Video(video_path)
table = wandb.Table(columns=["prompts", "negative_prompts", "interpolated_video"])
table.add_data(prompt, negative_prompt, video)
wandb.log({"Interpolated-Video": video, "Result-Table": table})

artifact = wandb.Artifact(name=f"video-{wandb.run.id}", type="video")
artifact.add_file(local_path=video_path)
wandb.log_artifact(artifact)
