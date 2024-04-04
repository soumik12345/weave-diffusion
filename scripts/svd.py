import os
from glob import glob

import torch
import wandb
from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from wandb.integration.diffusers import autolog

from weave_diffusion.utils import autogenerate_seed, get_generated_artifacts


autolog(init=dict(project="weave-diffusion", entity="geekyrakshit", job_type="final"))

generated_artifacts = get_generated_artifacts(
    "weave-diffusion", "geekyrakshit", "l6eyzpqa"
)
result_table_artifact_address = None
for artifact in generated_artifacts:
    if "ResultTablePipelineCall" in artifact:
        result_table_artifact_address = artifact
        break

artifact = wandb.use_artifact(result_table_artifact_address, type="run_table")
artifact_dir = artifact.download()
image_path = glob(os.path.join(artifact_dir, "media", "images", "*.png"))[0]
image = Image.open(image_path)

svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
svd_pipeline.enable_model_cpu_offload()
svd_pipeline.enable_attention_slicing()

height = 1920
width = 1080
generator_svd = torch.Generator(device="cuda").manual_seed(autogenerate_seed())

frames = svd_pipeline(
    image,
    height=height,
    width=width,
    num_frames=30,
    decode_chunk_size=8,
    generator=generator_svd,
).frames[0]
