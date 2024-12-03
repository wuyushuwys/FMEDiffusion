from typing import Optional, Tuple, Dict, Any, Union
import torch

# import model for animatediff
from diffusers.models.unets.unet_motion_model import (CrossAttnDownBlockMotion, DownBlockMotion, CrossAttnUpBlockMotion,
                                                      UpBlockMotion, UNetMidBlockCrossAttnMotion)
# import model for stable-video-diffusion
from diffusers.models.unets.unet_3d_blocks import (DownBlockSpatioTemporal, CrossAttnDownBlockSpatioTemporal,
                                                   UpBlockSpatioTemporal, CrossAttnUpBlockSpatioTemporal)

from diffusers.utils import logging, deprecate
from ..utils.config_utils import MemoryEfficientMixin, round_func

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

""" AnimateDiff Blocks """


class FMEDownBlockMotion(MemoryEfficientMixin, DownBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            num_frames: int = 1,
            *args,
            **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        output_states = ()

        blocks = zip(self.resnets, self.motion_modules)
        for resnet, motion_module in blocks:

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)

            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

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


class FMECrossAttnDownBlockMotion(MemoryEfficientMixin, CrossAttnDownBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            num_frames: int = 1,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            additional_residuals: Optional[torch.Tensor] = None,
    ):
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        output_states = ()
        blocks = list(zip(self.resnets, self.attentions, self.motion_modules))
        for i, (resnet, attn, motion_module) in enumerate(blocks):
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states = attn(
                    chunk_hidden_states,
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

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


class FMEUpBlockMotion(MemoryEfficientMixin, UpBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            upsample_size=None,
            num_frames: int = 1,
            *args,
            **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        blocks = zip(self.resnets, self.motion_modules)

        for resnet, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


class FMECrossAttnUpBlockMotion(MemoryEfficientMixin, CrossAttnUpBlockMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            res_hidden_states_tuple: Tuple[torch.Tensor, ...],
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            num_frames: int = 1,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        blocks = zip(self.resnets, self.attentions, self.motion_modules)
        for resnet, attn, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states = attn(
                    chunk_hidden_states,
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(0), chunk_size):
                    chunk_hidden_states = upsampler(hidden_states[i:i + chunk_size])
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


class FMEUNetMidBlockCrossAttnMotion(MemoryEfficientMixin, UNetMidBlockCrossAttnMotion):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            num_frames: int = 1,
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        blocks = zip(self.resnets[:-2], self.attentions, self.motion_modules)
        for resnet, attn, motion_module in blocks:
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = resnet(hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
                chunk_hidden_states = attn(
                    chunk_hidden_states,
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            if self.spatial_slicing:
                chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
                chunk_hidden_states_cache = []
                for i in range(0, hidden_states.size(-1), chunk_size):
                    chunk_hidden_states = motion_module(hidden_states[..., i:i + chunk_size], num_frames=num_frames)
                    if not torch.is_tensor(chunk_hidden_states):
                        chunk_hidden_states = chunk_hidden_states[0]
                    chunk_hidden_states_cache.append(chunk_hidden_states)
                hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)
            else:
                hidden_states = motion_module(hidden_states, num_frames=num_frames)
                if not torch.is_tensor(hidden_states):
                    hidden_states = hidden_states[0]

        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_hidden_states = self.resnets[-1](hidden_states[i:i + chunk_size], temb[i:i + chunk_size])
            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        return hidden_states


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
