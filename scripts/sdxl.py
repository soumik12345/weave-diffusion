import torch

from diffusers import StableVideoDiffusionPipeline, StableDiffusionXLPipeline
from wandb.integration.diffusers import autolog

from weave_diffusion.utils import autogenerate_seed


autolog(init=dict(project="weave-diffusion", entity="geekyrakshit", job_type="final"))

sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
sdxl_pipeline.enable_model_cpu_offload()
sdxl_pipeline.enable_attention_slicing()

prompt_1 = "a network of woolen balls connected by threads looking like cells, realistic rendering, color palette consisting of apricot, yello, cyan, pink, and white"
prompt_2 = ""
negative_prompt_1 = "metal, squirming, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality"
negative_prompt_2 = "metal, squirming, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality"
num_inference_steps = 50
guidance_scale = 5.0
height = 1920
width = 1080
sdxl_seed, svd_seed = autogenerate_seed(), autogenerate_seed()

# Make the experiment reproducible by controlling randomness.
# The seed would be automatically logged to WandB.
generator_sdxl = torch.Generator(device="cuda").manual_seed(sdxl_seed)
generator_svd = torch.Generator(device="cuda").manual_seed(svd_seed)

image = sdxl_pipeline(
    prompt=prompt_1,
    prompt_2=prompt_2,
    height=height,
    width=width,
    negative_prompt=negative_prompt_1,
    negative_prompt_2=negative_prompt_2,
    num_inference_steps=num_inference_steps,
    generator=generator_sdxl,
    guidance_scale=guidance_scale,
).images[0]

# sdxl_pipeline = sdxl_pipeline.to("cpu")
# del sdxl_pipeline
# gc.collect()
# torch.cuda.empty_cache()
