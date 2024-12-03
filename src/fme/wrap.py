import logging
from typing import Union

from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipeline
from diffusers.pipelines import AnimateDiffPipeline

from .scheme import AnimateDiffScheme, StableVideoDiffusionScheme
from .utils import MemoryEfficientMixin

logging.getLogger(__name__)


class FMEWrapper:

    def __init__(self, num_temporal_chunk, num_spatial_chunk=None, num_frames=None, verbose=False):
        self.num_temporal_chunk = num_temporal_chunk
        self.num_spatial_chunk = num_spatial_chunk
        self.verbose = verbose
        if num_frames is not None:
            if num_frames % num_temporal_chunk != 0:
                logging.warning(f"Please choose num_frames divisible by num_temporal_chunk for better performance."
                                f"Current get num_frames={num_frames} and num_temporal_chunk={num_temporal_chunk}.")

    def wrap(self,
             pipe: Union[StableVideoDiffusionPipeline, AnimateDiffPipeline,],
             ) -> None:
        # wrap animatediff
        if isinstance(pipe, AnimateDiffPipeline):
            for n, m in pipe.unet.named_modules():
                if isinstance(m, tuple(AnimateDiffScheme.keys())):
                    if self.verbose:
                        logging.info(f"{m.__class__.__name__} --> {AnimateDiffScheme[m.__class__].__name__}")
                    m.__class__ = AnimateDiffScheme[m.__class__]
                    self._assign_property(m)
            pipe.vae.use_slicing = True

        # wrap svd
        elif isinstance(pipe, StableVideoDiffusionPipeline):
            for n, m in pipe.unet.named_modules():
                if isinstance(m, tuple(StableVideoDiffusionScheme.keys())):
                    if self.verbose:
                        logging.info(f"{m.__class__.__name__} --> {StableVideoDiffusionScheme[m.__class__].__name__}")
                    m.__class__ = StableVideoDiffusionScheme[m.__class__]
                    self._assign_property(m)
            for n, m in pipe.vae.named_modules():
                if isinstance(m, tuple(StableVideoDiffusionScheme.keys())):
                    if self.verbose:
                        logging.info(f"{m.__class__.__name__} --> {StableVideoDiffusionScheme[m.__class__].__name__}")
                    m.__class__ = StableVideoDiffusionScheme[m.__class__]
                    self._assign_property(m)
        else:
            raise NotImplementedError(f"{pipe.__class__.__name__} is not implemented.")

    def _assign_property(self, m: MemoryEfficientMixin):
        m.num_chunk = self.num_temporal_chunk
        if self.num_spatial_chunk is not None:
            m.num_spatial_chunk = self.num_spatial_chunk
