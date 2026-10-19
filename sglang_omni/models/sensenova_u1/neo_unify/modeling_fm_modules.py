"""Compatibility exports for SenseNova flow heads and image decoders."""

from sglang_omni.models.sensenova_u1.neo_unify.flow_matching import (
    ConditionedFinalLayer,
    FinalLayer,
    FlowMatchingHead,
    ResBlock,
    RMSNorm,
    SimpleMLPAdaLN,
    TimestepConditionedMLP,
    TimestepEmbedder,
    modulate,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_decoders import ConvDecoder
from sglang_omni.models.sensenova_u1.neo_unify.image_decoders import (
    PatchDecoderPostPixelShuffle as PatchDecoder_postps,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_decoders import (
    PatchDecoderPrePixelShuffle as PatchDecoder_preps,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_decoders import (
    PatchDecoderPrePixelShuffleSingleProjection as PatchDecoder_preps1,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_decoders import (
    PostConvSmoother,
    ProgressiveConvDecoder,
    ResidualConvBlock,
)
from sglang_omni.models.sensenova_u1.neo_unify.position_embeddings import (
    NerfEmbedder,
    PositionEmbedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    interpolate_pos_embed,
    precompute_freqs_cis_2d,
)

# Modified for SGLang; see this directory's README.md for upstream source.


__all__ = [
    "ConditionedFinalLayer",
    "ConvDecoder",
    "FinalLayer",
    "FlowMatchingHead",
    "NerfEmbedder",
    "PatchDecoder_postps",
    "PatchDecoder_preps",
    "PatchDecoder_preps1",
    "PositionEmbedding",
    "PostConvSmoother",
    "ProgressiveConvDecoder",
    "RMSNorm",
    "ResBlock",
    "ResidualConvBlock",
    "SimpleMLPAdaLN",
    "TimestepConditionedMLP",
    "TimestepEmbedder",
    "get_1d_sincos_pos_embed_from_grid",
    "get_2d_sincos_pos_embed",
    "get_2d_sincos_pos_embed_from_grid",
    "interpolate_pos_embed",
    "modulate",
    "precompute_freqs_cis_2d",
]
