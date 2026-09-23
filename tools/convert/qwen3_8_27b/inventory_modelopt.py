"""Independent registered inventory for NVIDIA ModelOpt Qwen3.8-27B."""
from __future__ import annotations

from collections import Counter
from tools.convert.qwen3_6.common.inventory import (
    BF16, FP32, I32, Q4, Q5, Q6, W8, RESOURCE_SPECS, TensorSpec, ResourceSpec,
    CONTIGUOUS_LAYOUT, ROW_SPLIT_LAYOUT,
)
from tools.convert.qwen3_6.common.recipe import expression_shape
from . import inventory as dense_inventory
from . import recipe_modelopt as recipe

MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "nvfp4-modelopt"
TARGET_KEY = "qwen3_8_27b"
NVFP4 = "NVFP4_F32M"
FP8 = "FP8_E4M3FN_ROW_F32S"
BLOCK_SCALE_LAYOUT = "blockscale-k16-m128x4-multiplier-v1"
ROW_SCALE_LAYOUT = "row-scale-f32-v1"

def tensor_spec(name: str, shape: tuple[int, ...], format: str) -> TensorSpec:
    if format in (BF16, FP32, I32):
        layout = CONTIGUOUS_LAYOUT
    elif format == NVFP4:
        layout = BLOCK_SCALE_LAYOUT
    elif format == FP8:
        layout = ROW_SCALE_LAYOUT
    elif format in (Q4, Q5, Q6, W8):
        layout = ROW_SPLIT_LAYOUT
    else:
        raise ValueError(f"unsupported ModelOpt artifact format: {format}")
    return TensorSpec(name, shape, format, layout)

TEXT_CORE_TENSOR_SPECS = (
    tensor_spec("text/token_embedding", (248320, 5120), BF16),
    *(tensor_spec(r.object_name, expression_shape(r.expression),
                  FP32 if r.object_name.endswith(("/a_log", "/dt_bias")) else BF16)
      for r in recipe.DIRECT_RECIPES),
    *(tensor_spec(r.object_name, r.shape, FP8) for r in recipe.FP8_WEIGHT_RECIPES),
    *(tensor_spec(r.object_name, r.shape, NVFP4) for r in recipe.NVFP4_WEIGHT_RECIPES),
    *(tensor_spec(r.object_name, (), FP32) for r in recipe.INPUT_SCALE_RECIPES
      if not r.object_name.startswith("text/draft_head/")),
)
DRAFT_HEAD_TENSOR_SPECS = (
    tensor_spec("text/draft_head", (131072, 5120), NVFP4),
    tensor_spec("text/draft_head/input_scale_multiplier", (), FP32),
    tensor_spec("text/draft_head_token_ids", (131072,), I32),
)
# These are shared, immutable unquantized-source execution roles, not the old NVFP4 recipe.
MTP_TENSOR_SPECS = dense_inventory.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = dense_inventory.VISION_TENSOR_SPECS
TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS + DRAFT_HEAD_TENSOR_SPECS + MTP_TENSOR_SPECS + VISION_TENSOR_SPECS
OBJECT_SPECS = RESOURCE_SPECS + TENSOR_SPECS
FORMAT_COUNTS = dict(Counter(s.format for s in TENSOR_SPECS))
LAYOUT_COUNTS = dict(Counter(s.layout for s in TENSOR_SPECS))

def validate_inventory() -> None:
    if len({s.name for s in OBJECT_SPECS}) != len(OBJECT_SPECS):
        raise ValueError("duplicate ModelOpt object name")
    if (len(TEXT_CORE_TENSOR_SPECS), len(TENSOR_SPECS), len(OBJECT_SPECS)) != (916, 1264, 1270):
        raise ValueError("ModelOpt inventory is incomplete")
    if FORMAT_COUNTS != {BF16: 535, FP32: 354, I32: 1, Q4: 54, Q5: 54, Q6: 1, W8: 7,
                         NVFP4: 130, FP8: 128}:
        raise ValueError(f"unexpected ModelOpt format allocation: {FORMAT_COUNTS}")

validate_inventory()
