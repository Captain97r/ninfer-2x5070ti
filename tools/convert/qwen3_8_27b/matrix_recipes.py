"""Checkpoint-independent matrix row geometry for Qwen3.8-27B source adapters."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tools.convert.qwen3_6.common import recipe as family_recipe

FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))

@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class MatrixSource:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class MatrixPart:
    source: MatrixSource
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class Fp8WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]


@dataclass(frozen=True, slots=True)
class Nvfp4WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]
    scalar_sources: tuple[MatrixSource, ...]


@dataclass(frozen=True, slots=True)
class InputScaleRecipe:
    object_name: str
    sources: tuple[MatrixSource, ...]
    weight_names: tuple[str, ...]


def _source(name: str, n: int, k: int) -> MatrixSource:
    return MatrixSource(name, (n, k))


def _all(source: MatrixSource) -> MatrixPart:
    return MatrixPart(source, (RowRange(0, source.shape[0]),))


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    begin = 256 if gate else 0
    return MatrixPart(
        source,
        tuple(
            RowRange(head * 512 + begin, head * 512 + begin + 256)
            for head in range(24)
        ),
    )


def build_matrix_recipes(
    nvfp4_mlp_layers: tuple[int, ...], *, output_head_nvfp4: bool = False,
    input_scale_suffix: str = "input_scale_divisor",
) -> tuple[
    tuple[Fp8WeightRecipe, ...],
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputScaleRecipe, ...],
    tuple[tuple[MatrixSource, ...], ...],
]:
    fp8_weights: list[Fp8WeightRecipe] = []
    nvfp4_weights: list[Nvfp4WeightRecipe] = []
    input_scales: list[InputScaleRecipe] = []
    global_scale_groups: list[tuple[MatrixSource, ...]] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        (
                            _q_part(query, False),
                            _all(key),
                            _q_part(query, True),
                            _all(value),
                        ),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "attention/output",
                        output.shape,
                        (_all(output),),
                    ),
                )
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(
                source_prefix + "linear_attn.out_proj", 5120, 6144
            )
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        (16384, 5120),
                        (_all(query_key_value), _all(z)),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "gdn/output",
                        output.shape,
                        (_all(output),),
                    ),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        if layer in nvfp4_mlp_layers:
            gate_up_sources = (gate, up)
            global_scale_groups.append(gate_up_sources)
            nvfp4_weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "mlp/gate_up",
                        (34816, 5120),
                        (_all(gate), _all(up)),
                        gate_up_sources,
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "mlp/down",
                        down.shape,
                        (_all(down),),
                        (down,),
                    ),
                )
            )
            input_scales.extend(
                (
                    InputScaleRecipe(
                        object_prefix
                        + "mlp/gate_up_projection/" + input_scale_suffix,
                        gate_up_sources,
                        (object_prefix + "mlp/gate_up",),
                    ),
                    InputScaleRecipe(
                        object_prefix + "mlp/down_projection/" + input_scale_suffix,
                        (down,),
                        (object_prefix + "mlp/down",),
                    ),
                )
            )
        else:
            fp8_weights.extend(
                (
                    Fp8WeightRecipe(
                        object_prefix + "mlp/gate_up",
                        (34816, 5120),
                        (_all(gate), _all(up)),
                    ),
                    Fp8WeightRecipe(
                        object_prefix + "mlp/down",
                        down.shape,
                        (_all(down),),
                    ),
                )
            )

    output_head = _source("lm_head", 248320, 5120)
    if output_head_nvfp4:
        nvfp4_weights.append(Nvfp4WeightRecipe(
            "text/output_head", output_head.shape, (_all(output_head),), (output_head,)
        ))
        input_scales.append(InputScaleRecipe(
            "text/output_head/" + input_scale_suffix, (output_head,), ("text/output_head",)
        ))
    else:
        fp8_weights.append(Fp8WeightRecipe(
            "text/output_head", output_head.shape, (_all(output_head),)
        ))
    return (
        tuple(fp8_weights),
        tuple(nvfp4_weights),
        tuple(input_scales),
        tuple(global_scale_groups),
    )


def build_direct_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    recipes: list[family_recipe.TensorRecipe] = []
    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "input_norm",
                family_recipe.source(
                    source_prefix + "input_layernorm.weight", (5120,)
                ),
            )
        )
        if layer in FULL_ATTENTION_LAYERS:
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/query_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.q_norm.weight", (256,)
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/key_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.k_norm.weight", (256,)
                        ),
                    ),
                )
            )
        else:
            convolution = family_recipe.source(
                source_prefix + "linear_attn.conv1d.weight", (10240, 1, 4)
            )
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_log",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.A_log", (48,)
                            ),
                            "FP32",
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.dt_bias", (48,)
                            ),
                            "FP32",
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/convolution",
                        family_recipe.Transpose(
                            family_recipe.Reshape(
                                family_recipe.Slice(convolution, 1, 0, 1),
                                (10240, 4),
                            ),
                            (1, 0),
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_b_projection",
                        family_recipe.Concat(
                            (
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_a.weight",
                                    (48, 5120),
                                ),
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_b.weight",
                                    (48, 5120),
                                ),
                            ),
                            0,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/norm",
                        family_recipe.source(
                            source_prefix + "linear_attn.norm.weight", (128,)
                        ),
                    ),
                )
            )
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "post_attention_norm",
                family_recipe.source(
                    source_prefix + "post_attention_layernorm.weight", (5120,)
                ),
            )
        )
    recipes.append(
        family_recipe.TensorRecipe(
            "text/final_norm",
            family_recipe.source("model.language_model.norm.weight", (5120,)),
        )
    )
    return tuple(recipes)


def select_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, row_range.begin, row_range.rows)
        for row_range in part.rows
    ]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)
