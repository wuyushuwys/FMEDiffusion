from typing import Optional, Tuple
import numpy as np
import torch

from diffusers.models.resnet import SpatioTemporalResBlock
from diffusers.models.transformers.transformer_temporal import (TransformerSpatioTemporalModel,
                                                                TransformerTemporalModelOutput)
from diffusers.models.attention import TemporalBasicTransformerBlock, _chunked_feed_forward

from ..utils.config_utils import MemoryEfficientMixin, round_func


class FMESpatioTemporalResBlock(MemoryEfficientMixin, SpatioTemporalResBlock):

    def forward(
            self,
            hidden_states: torch.Tensor,
            temb: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_temb = temb[i:i + chunk_size] if temb is not None else temb
            chunk_hidden_states = self.spatial_res_block(hidden_states[i:i + chunk_size], chunk_temb)
            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        )

        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)

        chunk_size = max(round_func(hidden_states.size(-1) / self.num_spatial_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(-1), chunk_size):
            chunk_hidden_states = self.temporal_res_block(hidden_states[..., i:i + chunk_size], temb)
            chunk_hidden_states = self.time_mixer(
                x_spatial=hidden_states_mix[..., i:i + chunk_size],
                x_temporal=chunk_hidden_states,
                image_only_indicator=image_only_indicator,
            )
            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=-1)

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

        return hidden_states


class FMETemporalBasicTransformerBlock(MemoryEfficientMixin, TemporalBasicTransformerBlock):

    def forward(
            self,
            hidden_states: torch.Tensor,
            num_frames: int,
            encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_residual = hidden_states[i:i + chunk_size].cuda()

            chunk_hidden_states = self.norm_in(hidden_states[i:i + chunk_size].cuda())

            if self._chunk_size is not None:
                chunk_hidden_states = _chunked_feed_forward(self.ff_in, chunk_hidden_states, self._chunk_dim,
                                                            self._chunk_size)
            else:
                chunk_hidden_states = self.ff_in(chunk_hidden_states)

            if self.is_res:
                chunk_hidden_states = chunk_hidden_states + chunk_residual

            chunk_norm_hidden_states = self.norm1(chunk_hidden_states)
            chunk_attn_output = self.attn1(chunk_norm_hidden_states, encoder_hidden_states=None)
            chunk_hidden_states = chunk_attn_output + chunk_hidden_states

            # 3. Cross-Attention
            if self.attn2 is not None:
                chunk_norm_hidden_states = self.norm2(chunk_hidden_states)
                chunk_attn_output = self.attn2(chunk_norm_hidden_states,
                                               encoder_hidden_states=encoder_hidden_states[i:i + chunk_size])
                chunk_hidden_states = chunk_attn_output + chunk_hidden_states

            # 4. Feed-forward
            chunk_norm_hidden_states = self.norm3(chunk_hidden_states)

            if self._chunk_size is not None:
                chunk_ff_output = _chunked_feed_forward(self.ff, chunk_norm_hidden_states, self._chunk_dim,
                                                        self._chunk_size)
            else:
                chunk_ff_output = self.ff(chunk_norm_hidden_states)

            if self.is_res:
                chunk_hidden_states = chunk_ff_output + chunk_hidden_states
            else:
                chunk_hidden_states = chunk_ff_output

            chunk_hidden_states_cache.append(chunk_hidden_states)
        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)
        return hidden_states


class FMETransformerSpatioTemporalModel(MemoryEfficientMixin, TransformerSpatioTemporalModel):

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ):
        # 0. Device
        device = next(self.parameters()).device

        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0]
        time_context = time_context_first_timestep[None, :].broadcast_to(
            height * width, batch_size, 1, time_context.shape[-1]
        )
        time_context = time_context.reshape(height * width * batch_size, 1, time_context.shape[-1])

        residual = hidden_states

        chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
        chunk_hidden_states_cache = []
        for i in range(0, hidden_states.size(0), chunk_size):
            chunk_hidden_states = self.norm(hidden_states[i:i + chunk_size])
            inner_dim = chunk_hidden_states.shape[1]
            chunk_hidden_states = chunk_hidden_states.permute(0, 2, 3, 1).reshape(chunk_hidden_states.size(0),
                                                                                  height * width, inner_dim)
            chunk_hidden_states = self.proj_in(chunk_hidden_states)
            chunk_hidden_states_cache.append(chunk_hidden_states)

        hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            # spatio
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = block(
                    hidden_states[i:i + chunk_size],
                    encoder_hidden_states=encoder_hidden_states[i:i + chunk_size],
                )
                chunk_hidden_states_cache.append(chunk_hidden_states)

            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb.to(hidden_states_mix)

            # temporal
            # In-Block TemporalBasicTransformerBlock Slicing
            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )

            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            while hidden_states.shape[0] % chunk_size != 0:
                chunk_size += 1
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = self.time_mixer(
                    x_spatial=hidden_states[i:i + chunk_size],
                    x_temporal=hidden_states_mix[i:i + chunk_size],
                    image_only_indicator=image_only_indicator[..., i // batch_size:(i + chunk_size) // batch_size],
                )
                if len(self.transformer_blocks) == 1:
                    chunk_hidden_states = self.proj_out(chunk_hidden_states)
                chunk_hidden_states_cache.append(chunk_hidden_states)
            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)

        # 3. Output
        if len(self.transformer_blocks) != 1:
            chunk_size = max(round_func(hidden_states.size(0) / self.num_chunk), 1)
            chunk_hidden_states_cache = []
            for i in range(0, hidden_states.size(0), chunk_size):
                chunk_hidden_states = self.proj_out(hidden_states[i:i + chunk_size])
                chunk_hidden_states_cache.append(chunk_hidden_states)

            hidden_states = torch.cat(chunk_hidden_states_cache, dim=0)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
