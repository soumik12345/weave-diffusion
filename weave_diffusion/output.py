from dataclasses import dataclass
from typing import List

from PIL import Image

from diffusers.utils.outputs import BaseOutput


@dataclass
class StableDiffusionInterpolationOutput(BaseOutput):
    frames: List[Image.Image]
