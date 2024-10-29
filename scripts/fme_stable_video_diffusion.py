import logging
import time

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif

from fme import FMEWrapper

logging.basicConfig(level=logging.INFO)

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
pipe.to('cuda')

helper = FMEWrapper(num_temporal_chunk=7, num_spatial_chunk=7, num_frames=pipe.unet.config.num_frames)
helper.wrap(pipe)

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
time.sleep(2)

info = nvmlDeviceGetMemoryInfo(handle)
used = info.used
logging.info(f'[PyNVML]Peak Memory is: {used / (1024 ** 3):.02f}G')

export_to_gif(frames, "generated_fme.gif", fps=7)
