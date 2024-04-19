from typing import List, Optional

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableVideoDiffusionPipeline,
    AutoencoderKL,
)

import wandb
from wandb.integration.diffusers import autolog

import fire
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from weave_diffusion.utils import autogenerate_seed


controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(leave=False)

svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
svd_pipeline.enable_model_cpu_offload()
svd_pipeline.enable_attention_slicing()
svd_pipeline.set_progress_bar_config(leave=False)


def generate(
    prompts: List[str] = [
        "threads of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "threads of cotton on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "threads of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "threads of cotton on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "balls of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "balls of cotton on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "balls of wool on a black matte flat graphic design background, realistic rendering, color palette consisting of apricot, yellow, cyan, pink, and white, sharp, bright and contrasting colors",
        "yellow twine balls on a white fabric detailed graphic design background, realistic rendering, 4K, imax, unreal engine, sharp, bright and contrasting colors",
        "yellow cotton balls on a black matte flat graphic design background, realistic rendering, 4K, imax, unreal engine, sharp, bright and contrasting colors",
        "yellow woolen balls on a white matte flat graphic design background, realistic rendering, 4K, imax, logo design, sharp, bright and contrasting colors",
    ],
    negative_prompts: List[str] = [
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, blue, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
        "cross hatching, brown background, text, metal, apple, mango, orange, fruits, virus, bacteria, static, frame, painting, illustration, low quality, low resolution, greyscale, monochrome, cropped, lowres, jpeg artifacts, semi-realistic worst quality",
    ],
    condition_image_path="./logo_white.png",
    num_frames: int = 30,
    decode_chunk_size: int = 8,
    seed_sdxl: Optional[int] = None,
    seed_svd: Optional[int] = None,
):
    autolog(
        init=dict(
            project="weave-diffusion",
            entity="geekyrakshit",
            job_type="sdxl_controlnet+svd",
        )
    )
    condition_image = Image.open(condition_image_path)

    final_video_frames = []
    seeds_sdxl = (
        [autogenerate_seed() for _ in range(len(prompts))]
        if seed_sdxl is None
        else [seed_sdxl] * len(prompts)
    )
    seeds_svd = (
        [autogenerate_seed() for _ in range(len(prompts))]
        if seed_svd is None
        else [seed_svd] * len(prompts)
    )
    for idx, (prompt, negative_prompt) in tqdm(
        enumerate(zip(prompts, negative_prompts)), total=len(prompts)
    ):
        generator_sdxl = torch.Generator(device="cuda").manual_seed(seeds_sdxl[idx])
        generated_image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=condition_image,
            generator=generator_sdxl,
        ).images[0]
        generator_svd = torch.Generator(device="cuda").manual_seed(seeds_svd[idx])
        frames = svd_pipeline(
            generated_image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            generator=generator_svd,
            height=generated_image.size[1],
            width=generated_image.size[0],
        ).frames[0]
        frames = [np.array(frame) for frame in frames[::-1]]
        final_video_frames.extend(frames)

    final_video_frames = np.transpose(np.array(final_video_frames), (0, 3, 1, 2))
    wandb.log({"Generated-Video/Final": wandb.Video(final_video_frames, fps=30)})


if __name__ == "__main__":
    fire.Fire(generate)
