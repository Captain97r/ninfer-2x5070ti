"""Closed, source-word-preserving NVIDIA ModelOpt Qwen3.8-27B recipe."""
from __future__ import annotations

import struct
from typing import Mapping
import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.artifact.layouts import encode_direct
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import recipe as dense_recipe
from . import inventory as dense_inventory
from .matrix_recipes import (
    Fp8WeightRecipe, Nvfp4WeightRecipe, MatrixSource, InputScaleRecipe,
    build_matrix_recipes, build_direct_recipes, select_rows,
)

SOURCE_REPOSITORY = "nvidia/Qwen3.8-27B-NVFP4"
SOURCE_REVISION = "482ca0f3832238542f8f5295dde86b5f22711d80"
FP8_WEIGHT_RECIPES, NVFP4_WEIGHT_RECIPES, _nv_inputs, _ = build_matrix_recipes(
    tuple(range(64)), output_head_nvfp4=True, input_scale_suffix="input_scale_multiplier"
)
FP8_WEIGHTS_BY_NAME = {x.object_name: x for x in FP8_WEIGHT_RECIPES}
NVFP4_WEIGHTS_BY_NAME = {x.object_name: x for x in NVFP4_WEIGHT_RECIPES}
FP8_SOURCES = tuple(dict.fromkeys(p.source for r in FP8_WEIGHT_RECIPES for p in r.parts))
NVFP4_SOURCES = tuple(dict.fromkeys(p.source for r in NVFP4_WEIGHT_RECIPES for p in r.parts))
DIRECT_RECIPES = build_direct_recipes()
DENSE_RECIPES = DIRECT_RECIPES + tuple(
    dense_recipe.RECIPES_BY_NAME[s.name]
    for s in (*dense_inventory.MTP_TENSOR_SPECS, *dense_inventory.VISION_TENSOR_SPECS)
)
DENSE_BY_NAME = {r.object_name: r for r in DENSE_RECIPES}
EMBEDDING_SOURCE = family_recipe.source("model.language_model.embed_tokens.weight", (248320, 5120))

def _fp8_input(r: Fp8WeightRecipe) -> InputScaleRecipe:
    parent, leaf = r.object_name.rsplit("/", 1)
    site = "output_projection" if leaf == "output" else "input_projection"
    return InputScaleRecipe(
        parent + "/" + site + "/input_scale_multiplier",
        tuple(dict.fromkeys(p.source for p in r.parts)), (r.object_name,)
    )

INPUT_SCALE_RECIPES = (
    tuple(_fp8_input(r) for r in FP8_WEIGHT_RECIPES) + _nv_inputs +
    (InputScaleRecipe("text/draft_head/input_scale_multiplier",
                      (MatrixSource("lm_head", (248320, 5120)),), ("text/draft_head",)),)
)
INPUT_SCALES_BY_NAME = {r.object_name: r for r in INPUT_SCALE_RECIPES}

def source_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    dense = (*DENSE_RECIPES, family_recipe.TensorRecipe("text/token_embedding", EMBEDDING_SOURCE))
    out = {s.name: (s.shape, s.dtype) for s in family_recipe.source_requirements(dense).values()}
    for s in FP8_SOURCES:
        out[s.field("weight")] = (s.shape, "F8_E4M3")
        for field in ("weight_scale", "input_scale"):
            out[s.field(field)] = ((), "F32")
    for s in NVFP4_SOURCES:
        n, k = s.shape
        out[s.field("weight")] = ((n, k // 2), "U8")
        out[s.field("weight_scale")] = ((n, k // 16), "F8_E4M3")
        for field in ("weight_scale_2", "input_scale"):
            out[s.field(field)] = ((), "F32")
    return out

SOURCE_REQUIREMENTS = source_requirements()

def validate_metadata(metadata: Mapping[str, object]) -> family_recipe.SourcePreflight:
    if set(metadata) != set(SOURCE_REQUIREMENTS):
        missing = sorted(set(SOURCE_REQUIREMENTS) - set(metadata))
        unexpected = sorted(set(metadata) - set(SOURCE_REQUIREMENTS))
        raise ValueError(f"ModelOpt source tensor inventory mismatch: missing={missing[:1]}, unexpected={unexpected[:1]}")
    counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, (shape, dtype) in SOURCE_REQUIREMENTS.items():
        m = metadata[name]
        if (m.shape, m.dtype) != (shape, dtype):
            raise ValueError(f"{name}: ModelOpt source signature {(m.shape, m.dtype)} != {(shape, dtype)}")
        counts[dtype] = counts.get(dtype, 0) + 1
        shards.add(m.shard)
    return family_recipe.SourcePreflight(
        recipe_count=len(DENSE_RECIPES) + len(FP8_WEIGHT_RECIPES) + len(NVFP4_WEIGHT_RECIPES),
        source_tensor_count=len(metadata), source_shard_count=len(shards), source_dtype_counts=counts
    )

def scalar_word(tensor: torch.Tensor, name: str) -> bytes:
    if tensor.dtype != torch.float32 or tensor.shape != torch.Size([]):
        raise ValueError(f"{name}: expected scalar FP32")
    raw = tensor.detach().cpu().contiguous().view(torch.int32).item() & 0xffffffff
    if not valid_positive_fp32_word(raw):
        raise ValueError(f"{name}: multiplier must be finite and positive")
    return struct.pack("<I", raw)

def same_scalar(reader: ShardReader, sources: tuple[MatrixSource, ...], field: str) -> bytes:
    values = [scalar_word(reader.get(s.field(field)), s.field(field)) for s in sources]
    if not values or any(x != values[0] for x in values[1:]):
        raise ValueError(f"{sources[0].name if sources else field}: fused {field} words differ")
    return values[0]

def preflight_source(reader: ShardReader) -> family_recipe.SourcePreflight:
    result = validate_metadata(reader.metadata(reader.names))
    # Validate all 802 calibration words before opening the output writer.
    for name, (_, dtype) in SOURCE_REQUIREMENTS.items():
        if dtype == "F32":
            scalar_word(reader.get(name), name)
    for r in INPUT_SCALE_RECIPES:
        same_scalar(reader, r.sources, "input_scale")
    for r in NVFP4_WEIGHT_RECIPES:
        same_scalar(reader, r.scalar_sources, "weight_scale_2")
    return result

def materialize_fp8_weight(r: Fp8WeightRecipe, reader: ShardReader) -> tuple[torch.Tensor, torch.Tensor]:
    codes, scales = [], []
    for p in r.parts:
        s = p.source
        w = reader.get(s.field("weight"))
        if w.dtype != torch.float8_e4m3fn or tuple(w.shape) != s.shape:
            raise ValueError(f"{s.name}: FP8 weight signature mismatch")
        scale = same_scalar(reader, (s,), "weight_scale")
        scalar = torch.frombuffer(bytearray(scale), dtype=torch.float32)
        codes.append(select_rows(w.view(torch.uint8), p))
        scales.append(scalar.expand(p.output_rows).clone())
    return torch.cat(codes).contiguous(), torch.cat(scales).contiguous()

def materialize_nvfp4_weight(r: Nvfp4WeightRecipe, reader: ShardReader) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed, scales = [], []
    for p in r.parts:
        n, k = p.source.shape
        w, scale = reader.get(p.source.field("weight")), reader.get(p.source.field("weight_scale"))
        if w.dtype != torch.uint8 or tuple(w.shape) != (n, k // 2):
            raise ValueError(f"{p.source.name}: packed weight signature mismatch")
        if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != (n, k // 16):
            raise ValueError(f"{p.source.name}: block-scale signature mismatch")
        packed.append(select_rows(w, p))
        scales.append(select_rows(scale.view(torch.uint8), p))
    return (torch.cat(packed).contiguous(), torch.cat(scales).contiguous(),
            same_scalar(reader, r.scalar_sources, "weight_scale_2"))

def materialize_draft_head(reader: ShardReader, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    """Select packed rows, not decoded/requantized approximations of the target head."""
    source = NVFP4_WEIGHTS_BY_NAME["text/output_head"].parts[0].source
    if selected.dtype != torch.int64 or selected.ndim != 1:
        raise ValueError("draft shortlist must be a one-dimensional int64 tensor")
    if selected.numel() == 0 or int(selected.min()) < 0 or int(selected.max()) >= 248077:
        raise ValueError("draft shortlist exceeds tokenizer domain")
    if selected.unique().numel() != selected.numel():
        raise ValueError("draft shortlist contains duplicate IDs")
    # Head rows are already packed in the checkpoint. Select from their source
    # views directly: never construct FP32 weights or a full extra packed head.
    n, k = source.shape
    packed = reader.get(source.field("weight"))
    scales = reader.get(source.field("weight_scale"))
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (n, k // 2):
        raise ValueError("lm_head: packed weight signature mismatch")
    if scales.dtype != torch.float8_e4m3fn or tuple(scales.shape) != (n, k // 16):
        raise ValueError("lm_head: block-scale signature mismatch")
    return (packed.index_select(0, selected), scales.view(torch.uint8).index_select(0, selected),
            same_scalar(reader, (source,), "weight_scale_2"))


def iter_embedding_payload(reader: ShardReader, shape: tuple[int, int], *, rows_per_chunk: int = 1024):
    """Stream original BF16 words, with no whole-matrix cast or byte copy."""
    if type(rows_per_chunk) is not int or rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be a positive integer")
    weight = reader.get(EMBEDDING_SOURCE.name)
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != shape:
        raise ValueError("embedding: BF16 source signature mismatch")
    for begin in range(0, shape[0], rows_per_chunk):
        yield encode_direct(weight[begin:begin + rows_per_chunk], "BF16")
