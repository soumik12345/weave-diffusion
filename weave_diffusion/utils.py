import base64
import os
import random
import time
from typing import List, Tuple

import cv2
import imageio
import numpy as np
import torch
import wandb
from PIL import Image


def slerp(v0, v1, num, t0=0, t1=1):
    v0 = v0.detach().cpu().numpy()
    v1 = v1.detach().cpu().numpy()

    def interpolation(t, v0, v1, DOT_THRESHOLD=0.9995):
        """helper function to spherically interpolate two arrays v1 v2"""
        dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
        if np.abs(dot) > DOT_THRESHOLD:
            v2 = (1 - t) * v0 + t * v1
        else:
            theta_0 = np.arccos(dot)
            sin_theta_0 = np.sin(theta_0)
            theta_t = theta_0 * t
            sin_theta_t = np.sin(theta_t)
            s0 = np.sin(theta_0 - theta_t) / sin_theta_0
            s1 = sin_theta_t / sin_theta_0
            v2 = s0 * v0 + s1 * v1
        return v2

    t = np.linspace(t0, t1, num)
    v3 = torch.tensor(np.array([interpolation(t[i], v0, v1) for i in range(num)]))
    return v3


def autogenerate_seed() -> int:
    max_seed = int(1024 * 1024 * 1024)
    seed = random.randint(1, max_seed)
    seed = -seed if seed < 0 else seed
    seed = seed % max_seed
    return seed


def log_video(images: List[Image.Image], save_path: str) -> str:
    os.makedirs(save_path, exist_ok=True)
    filename = time.strftime("%H:%M:%S", time.localtime()).replace(":", "-")
    save_file_path = os.path.join(save_path, f"{filename}.mp4")
    with imageio.get_writer(save_file_path, fps=10) as video:
        for image in images:
            video.append_data(np.array(image))
    return save_file_path


def get_generated_artifacts(project: str, entity: str, run_id: str) -> Tuple[str, str]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    output_artifacts = run.logged_artifacts()
    return [f"{entity}/{project}/{artifact.name}" for artifact in output_artifacts]


def get_images_from_run(project: str, entity: str, run_id: str) -> str:
    api = wandb.Api()
    run = api.run("<entity>/<project>/<run_id>")
    for file in run.files():
        if file.name.endswith(".png"):
            file.download()
            return file.name


def pad_image(image, target_shape):
    original_shape = image.shape
    vertical_padding = (target_shape[0] - original_shape[0]) // 2
    extra_vertical_padding = (target_shape[0] - original_shape[0]) % 2
    horizontal_padding = (target_shape[1] - original_shape[1]) // 2
    extra_horizontal_padding = (target_shape[1] - original_shape[1]) % 2
    padded_image = np.pad(
        image,
        pad_width=(
            (vertical_padding, vertical_padding + extra_vertical_padding),
            (horizontal_padding, horizontal_padding + extra_horizontal_padding),
            (0, 0),
        ),
        mode="constant",
        constant_values=0,
    )
    return padded_image


def convert_to_canny(image, canny_low_threshold, canny_high_threshold):
    image = cv2.Canny(image, canny_low_threshold, canny_high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def center_crop(img_array, crop_percentage):
    crop_percent = crop_percentage / 100.0
    height, width, _ = img_array.shape
    crop_size = min(height, width) * crop_percent
    start_x = (width - crop_size) // 2
    end_x = start_x + crop_size
    start_y = (height - crop_size) // 2
    end_y = start_y + crop_size
    start_x, end_x, start_y, end_y = int(start_x), int(end_x), int(start_y), int(end_y)
    cropped_img = img_array[start_y:end_y, start_x:end_x]
    return cropped_img


def get_base64_string_from_image_file(image_file: str):
    with open(image_file, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
