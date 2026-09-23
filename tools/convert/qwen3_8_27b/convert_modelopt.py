"""Import NVIDIA ModelOpt Qwen3.8-27B without requantizing packed source weights.

python3 -m tools.convert.qwen3_8_27b.convert_modelopt --model <checkpoint> \
  --out <directory>/qwen3_8_27b_nvfp4_modelopt.ninfer --device cpu
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.artifact.layouts import encode_direct, encode_fp8_row_scaled, encode_nvfp4
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config, draft_head
from . import frontend_modelopt as frontend_converter
from . import inventory_modelopt as inventory, recipe_modelopt as recipe

RECIPE_ID = "qwen3_8_27b_nvfp4_modelopt-v1"
OUTPUT_BASENAME = "qwen3_8_27b_nvfp4_modelopt.ninfer"

@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: family_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    summary = family_config.validate_config(config)
    q = config.get("quantization_config")
    if not isinstance(q, dict):
        raise ValueError("ModelOpt source requires quantization_config")
    family_conversion.check_members("quantization_config", q, {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "ignore": ["mtp*", "mtp.layers.0*"],
    })
    producer = q.get("producer")
    if not isinstance(producer, dict) or producer.get("name") != "modelopt":
        raise ValueError("quantization producer must be modelopt")
    expected = {s.name: {"quant_algo": "FP8"} for s in recipe.FP8_SOURCES}
    expected.update({s.name: {"quant_algo": "NVFP4", "group_size": 16}
                     for s in recipe.NVFP4_SOURCES})
    if q.get("quantized_layers") != expected:
        raise ValueError("ModelOpt quantized_layers does not match the registered allocation")
    groups = q.get("config_groups")
    if not isinstance(groups, dict) or set(groups) != {"group_0", "group_1"}:
        raise ValueError("ModelOpt config requires exactly group_0 and group_1")
    for key, sources, bits in (("group_0", recipe.FP8_SOURCES, 8),
                               ("group_1", recipe.NVFP4_SOURCES, 4)):
        g = groups[key]
        if not isinstance(g, dict) or not isinstance(g.get("targets"), list):
            raise ValueError(f"{key}: malformed quantization group")
        targets = g["targets"]
        if len(targets) != len(sources) or set(targets) != {s.name for s in sources}:
            raise ValueError(f"{key}: quantization targets differ")
        required = {"dynamic": False, "num_bits": bits, "type": "float"}
        if bits == 4:
            required["group_size"] = 16
        for field in ("weights", "input_activations"):
            value = g.get(field)
            if not isinstance(value, dict):
                raise ValueError(f"{key}.{field}: missing quantization metadata")
            family_conversion.check_members(f"{key}.{field}", value, required)
    return summary

def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    index = family_conversion.load_json(model / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint tensor index must contain weight_map")
    shards = set(weight_map.values())
    if any(not isinstance(s, str) or Path(s).name != s for s in shards):
        raise ValueError("checkpoint index shard names must be local filenames")
    if shards != {p.name for p in model.glob("*.safetensors")}:
        raise ValueError("checkpoint shards do not match its index")
    summary = validate_config(family_conversion.load_json(model / "config.json"))
    inventory.validate_inventory()
    with ShardReader(model) as reader:
        source = recipe.preflight_source(reader)
    resources = frontend_converter.load_resources(model)
    object_plan = family_conversion.build_object_plan(
        inventory.OBJECT_SPECS, {r.name: r.data for r in resources}
    )
    draft = frontend_converter.compute_shortlist(
        _repo_root() / draft_head.DEFAULT_RANKING, model, resources
    )
    return ConversionPreflight(model, summary, source, resources, draft, object_plan)

def encode_object(spec: inventory.TensorSpec, reader: ShardReader,
                  draft: draft_head.DraftHeadContext):
    if spec.name == "text/token_embedding":
        return recipe.iter_embedding_payload(reader, spec.shape)
    if spec.name == "text/draft_head_token_ids":
        return encode_direct(draft_head.materialize_draft_head_token_ids(draft), "I32")
    if spec.name in recipe.INPUT_SCALES_BY_NAME:
        r = recipe.INPUT_SCALES_BY_NAME[spec.name]
        return recipe.same_scalar(reader, r.sources, "input_scale")
    if spec.name in recipe.FP8_WEIGHTS_BY_NAME:
        codes, scales = recipe.materialize_fp8_weight(recipe.FP8_WEIGHTS_BY_NAME[spec.name], reader)
        return encode_fp8_row_scaled(codes, scales, spec.shape, format=inventory.FP8)
    if spec.name == "text/draft_head":
        packed, scales, global_scale = recipe.materialize_draft_head(
            reader, torch.from_numpy(draft.selected)
        )
        return encode_nvfp4(packed, scales, global_scale, spec.shape, format=inventory.NVFP4)
    if spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
        packed, scales, global_scale = recipe.materialize_nvfp4_weight(
            recipe.NVFP4_WEIGHTS_BY_NAME[spec.name], reader
        )
        return encode_nvfp4(packed, scales, global_scale, spec.shape, format=inventory.NVFP4)
    value = family_recipe.materialize_recipe(recipe.DENSE_BY_NAME[spec.name], reader)
    if tuple(value.shape) != spec.shape:
        raise ValueError(f"{spec.name}: materialized shape does not match inventory")
    return family_conversion.encode_tensor_payload(value, spec, torch.device("cpu"))

def convert(model_dir: str | Path, out_path: str | Path, *, device: str = "cpu") -> Path:
    if device != "cpu":
        raise ValueError("ModelOpt import currently runs explicitly on CPU; use --device cpu")
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"ModelOpt converter output basename must be {OUTPUT_BASENAME!r}")
    started = time.perf_counter()
    preflight = preflight_conversion(model_dir)
    print(f"preflight complete: {len(preflight.object_plan.objects)} objects; CPU conversion", flush=True)
    resources = {r.name: r.data for r in preflight.resources}
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    with ShardReader(preflight.model_dir) as reader, ArtifactWriter(
        output, identity, preflight.object_plan.specs
    ) as writer:
        if writer.objects != preflight.object_plan.objects:
            raise RuntimeError("writer plan differs from preflight")
        for i, spec in enumerate(inventory.OBJECT_SPECS, 1):
            payload = (resources[spec.name] if isinstance(spec, inventory.ResourceSpec)
                       else encode_object(spec, reader, preflight.draft))
            writer.write(spec.name, payload)
            del payload
            print(f"[{i}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)
    report = family_conversion.build_conversion_report(
        identity=identity, target_key=inventory.TARGET_KEY, recipe_id=RECIPE_ID,
        repo_root=_repo_root(), model_dir=preflight.model_dir, out_path=output,
        arguments={"model": str(model_dir), "out": str(out_path), "device": "cpu"},
        config_summary=preflight.config_summary, source_preflight=preflight.source,
        objects=preflight.object_plan.objects, elapsed_seconds=time.perf_counter()-started,
        final_bytes=output.stat().st_size, device=torch.device("cpu"),
        ranking_path=preflight.draft.ranking,
    )
    report["source"].update(repository=recipe.SOURCE_REPOSITORY, revision=recipe.SOURCE_REVISION)
    report["packed_weight_import"] = {
        "fp8": "exact E4M3FN words and FP32 scalar multipliers replicated per fused row",
        "nvfp4": "exact E2M1 packed words, E4M3FN block scales and FP32 global multipliers",
        "input_scales": "exact scalar FP32; fused-source equality checked before writing",
        "draft_head": "exact packed lm_head rows/scales in the registered shortlist order",
    }
    report["frontend_normalization"] = {
        "resources": list(frontend_converter.NORMALIZED_RESOURCES),
        "contract": "canonical registered Qwen3.8 metadata reconstructed from identical source vocabulary/template; generation semantics unchanged",
    }
    report["derived_encoders"] = {
        "embedding": "none; exact original BF16 words",
        "mtp_and_vision": "existing registered groupwise encoders from this checkpoint's BF16 tensors",
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"complete: {output.stat().st_size} bytes; report={report_path}", flush=True)
    return report_path

def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    a = p.parse_args(argv)
    convert(a.model, a.out, device=a.device)

if __name__ == "__main__":
    main()
