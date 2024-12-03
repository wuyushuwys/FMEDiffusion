import logging

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName

from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

logging.basicConfig(level=logging.INFO)

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

pipe.to('cuda')

logging.info(f'Benchmark Peak Memory')

info = nvmlDeviceGetMemoryInfo(handle)
used = info.used
torch.cuda.reset_peak_memory_stats()
logging.info(f'[PyNVML]{nvmlDeviceGetName(handle)} Model Load Memory is: {used / (1024 ** 3):.02f}G')

# for benchmark memory
frames = pipe(
    prompt="raccoon playing a guitar",
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    height=576,
    width=1024,
    guidance_scale=7.5,
    num_inference_steps=1,
    generator=torch.Generator("cpu").manual_seed(42),
).frames[0]

info = nvmlDeviceGetMemoryInfo(handle)
used = info.used
logging.info(f'[PyNVML]{nvmlDeviceGetName(handle)} Peak Memory is: {used / (1024 ** 3):.02f}G')

logging.info(f'Inference')

frames = pipe(
    prompt="raccoon playing a guitar",
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    height=576,
    width=1024,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
).frames[0]

export_to_gif(frames, "generated_animatediff.gif", fps=7)
