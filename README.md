# Fast and Memory-Efficient Video Diffusion Using Streamlined Inference

Official Implementation of **NeurIPS2024** Fast and Memory-Efficient Video Diffusion Using Streamlined Inference

> **Fast and Memory-Efficient Video Diffusion using Streamlined Inference**   
> [Zheng Zhan*](https://zhanzheng8585.github.io), [Yushu Wu*](https://scholar.google.com/citations?user=3hEDsFYAAAAJ&hl=en), [Yifan Gong](https://yifanfanfanfan.github.io/), Zichong Meng, [Zhenglun Kong](https://zlkong.github.io/homepage/), Changdi Yang, Geng Yuan, Puzhao, Wei Nui, and Yanzhi Wang  
> Northeastern University, Harvard University, University of Georgia  
> 38th Conference on Neural Information Processing Systems ([**NeurIPS 2024**](https://neurips.cc/Conferences/2024/))

---
This repo contains simulation of **_Feature Slicer (Sec.4.1)_** and **_Operator Grouping (Sec.4.2)_** which can effectively reduce the memory-footprint of spatial-temporal model in inference.

---


## Dependencies
```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install pynvml  # for memory-footprint benchmark
pip install .
```
---
## Usage
```python
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif
# import our module wrapper
from fme import FMEWrapper

# load pipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
pipe.to('cuda')

# initialize wrapper
helper = FMEWrapper(num_temporal_chunk=7, num_spatial_chunk=7, num_frames=pipe.unet.config.num_frames)
# wrap pipeline
helper.wrap(pipe)

# Inference as normal
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

generator = torch.manual_seed(42)
# no decode_chunk_size required!
frames = pipe(image, generator=generator).frames[0]

export_to_gif(frames, "generated_fme.gif", fps=7)
```

---
## Note

#### The peak memory values may not exactly match those reported in the paper.
- In the case of SVD (`num_frames=14`, resolution=`576x1024`), the original peak memory reported in the paper is 39.49 GB, which can be reduced to 23.42 GB using our proposed method.
However, using the [example](scripts/fme_stable_video_diffusion.py), you may observe a peak memory of around 24.49 GB using our method, and note that the original peak memory could also rise to 40.39 GB.
These values may differ slightly from those reported in the paper.
