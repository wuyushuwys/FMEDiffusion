import logging
from typing import Union

from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
from diffusers.pipelines import AnimateDiffPipeline

from .scheme import StableVideoDiffusionScheme
from .utils import MemoryEfficientMixin

logging.getLogger(__name__)


class FMEWrapper:

    def __init__(self, num_temporal_chunk, num_spatial_chunk=None, num_frames=None):
        self.num_spatial_chunk = num_spatial_chunk
        self.num_temporal_chunk = num_temporal_chunk
        if num_frames is not None:
            if num_frames % num_temporal_chunk != 0:
                logging.warning(f"Please choose num_frames divisible by num_temporal_chunk for better performance."
                                f"Current get num_frames={num_frames} and num_temporal_chunk={num_temporal_chunk}.")
        else:
            logging.warning(f"Please enter num_frames for configuration check to enable better performance.")

    def wrap(self,
             pipe: Union[StableVideoDiffusionPipeline, AnimateDiffPipeline,],
             ) -> None:
        # wrap animatediff
        if isinstance(pipe, AnimateDiffPipeline):
            raise NotImplementedError

        # wrap svd
        elif isinstance(pipe, StableVideoDiffusionPipeline):
            for n, m in pipe.unet.named_modules():
                if isinstance(m, tuple(StableVideoDiffusionScheme.keys())):
                    m.__class__ = StableVideoDiffusionScheme[m.__class__]
                    self._assign_property(m)
            for n, m in pipe.vae.named_modules():
                if isinstance(m, tuple(StableVideoDiffusionScheme.keys())):
                    m.__class__ = StableVideoDiffusionScheme[m.__class__]
                    self._assign_property(m)
        else:
            raise NotImplementedError

    def _assign_property(self, m: MemoryEfficientMixin):
        m.num_chunk = self.num_temporal_chunk
        if self.num_spatial_chunk is not None:
            m.num_spatial_chunk = self.num_spatial_chunk
