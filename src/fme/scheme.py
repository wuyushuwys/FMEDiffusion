# stable-video diffusion
from diffusers.models.resnet import SpatioTemporalResBlock
from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel
from diffusers.models.attention import TemporalBasicTransformerBlock
from .nn.blocks import (FMESpatioTemporalResBlock,
                        FMETransformerSpatioTemporalModel,
                        FMETemporalBasicTransformerBlock)

from diffusers.models.unets.unet_3d_blocks import (DownBlockSpatioTemporal,
                                                   CrossAttnDownBlockSpatioTemporal,
                                                   UpBlockSpatioTemporal,
                                                   CrossAttnUpBlockSpatioTemporal)
from diffusers.models.unets import UNetSpatioTemporalConditionModel
from .models import FMEUNetSpatioTemporalConditionModel
from .nn.unet_blocks import (FMEDownBlockSpatioTemporal,
                             FMECrossAttnDownBlockSpatioTemporal,
                             FMEUpBlockSpatioTemporal,
                             FMECrossAttnUpBlockSpatioTemporal)

__all__ = ["StableVideoDiffusionScheme", ]
# animatediff scheme

# todo: animatediff

# stable-video diffusion scheme
StableVideoDiffusionScheme = {
    UNetSpatioTemporalConditionModel: FMEUNetSpatioTemporalConditionModel,
    SpatioTemporalResBlock: FMESpatioTemporalResBlock,  # for UNet and Temporal-Decoder
    TransformerSpatioTemporalModel: FMETransformerSpatioTemporalModel,
    TemporalBasicTransformerBlock: FMETemporalBasicTransformerBlock,
    DownBlockSpatioTemporal: FMEDownBlockSpatioTemporal,
    CrossAttnDownBlockSpatioTemporal: FMECrossAttnDownBlockSpatioTemporal,
    UpBlockSpatioTemporal: FMEUpBlockSpatioTemporal,
    CrossAttnUpBlockSpatioTemporal: FMECrossAttnUpBlockSpatioTemporal,
}
