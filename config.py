from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class Config:
    temp_image_dir:str = "iterations"
    gif_dir:str = "amimated"