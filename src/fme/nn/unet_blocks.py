from typing import Optional, Tuple
import torch

# import model for animatediff
from diffusers.models.unets.unet_3d_blocks import (CrossAttnUpBlockMotion, DownBlockMotion, CrossAttnUpBlockMotion,
                                                   UpBlockMotion)
# import model for stable-video-diffusion
from diffusers.models.unets.unet_3d_blocks import (DownBlockSpatioTemporal, CrossAttnDownBlockSpatioTemporal,
                                                   UpBlockSpatioTemporal, CrossAttnUpBlockSpatioTemporal)

from ..utils.config_utils import MemoryEfficientMixin, round_func

""" AnimateDiff Blocks """

# todo: animatediff block

""" Stable-Video Diffusion Blocks """


class FMEDownBlockSpatioTemporal(MemoryEfficientMixin, DownBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class FMECrossAttnDownBlockSpatioTemporal(MemoryEfficientMixin, CrossAttnDownBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = downsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            output_states = output_states + (hidden_states,)
        return hidden_states, output_states


class FMEUpBlockSpatioTemporal(MemoryEfficientMixin, UpBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


class FMECrossAttnUpBlockSpatioTemporal(MemoryEfficientMixin, CrossAttnUpBlockSpatioTemporal):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states
