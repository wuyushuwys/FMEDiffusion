import logging

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif

logging.basicConfig(level=logging.INFO)

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
pipe.to('cuda')

info = nvmlDeviceGetMemoryInfo(handle)
used = info.used
torch.cuda.reset_peak_memory_stats()
logging.info(f'[PyNVML] Model Load Memory is: {used / (1024 ** 3):.02f}G')

# Load the conditioning image
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
frames = pipe(image, generator=generator).frames[0]

info = nvmlDeviceGetMemoryInfo(handle)
used = info.used
logging.info(f'[PyNVML]Peak Memory is: {used / (1024 ** 3):.02f}G')

export_to_gif(frames, "generated.gif", fps=7)
