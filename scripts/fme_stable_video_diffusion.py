import logging
import time

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName

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

logging.info(f'Benchmark Peak Memory')

helper = FMEWrapper(num_temporal_chunk=7, num_spatial_chunk=7, num_frames=pipe.unet.config.num_frames)
helper.wrap(pipe)

info = nvmlDeviceGetMemoryInfo(handle)
used = info.used
torch.cuda.reset_peak_memory_stats()
logging.info(f'[PyNVML]{nvmlDeviceGetName(handle)} Model Load Memory is: {used / (1024 ** 3):.02f}G')

# Load the conditioning image
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

# for benchmark memory
frames = pipe(image, generator=torch.manual_seed(42), num_inference_steps=1, height=576, width=1024).frames[0]

info = nvmlDeviceGetMemoryInfo(handle)
used = info.used
logging.info(f'[PyNVML]{nvmlDeviceGetName(handle)} Peak Memory is: {used / (1024 ** 3):.02f}G')

logging.info(f'Inference')

frames = pipe(image, generator=torch.manual_seed(42)).frames[0]

export_to_gif(frames, "generated_fme_svd.gif", fps=7)
