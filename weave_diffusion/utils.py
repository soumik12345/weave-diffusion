import os
import random
import time
from typing import List

import imageio
import numpy as np
import torch
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
