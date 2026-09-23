from __future__ import annotations

from dataclasses import replace
import json
import struct

import pytest
import torch

from tools.artifact.layouts import encode_direct, encoded_size
from tools.convert.common.safetensors import TensorMetadata
from tools.convert.qwen3_8_27b import (
    convert_modelopt as converter, inventory_modelopt as inventory,
    recipe_modelopt as recipe, frontend_modelopt as frontend,
)
from tools.convert.qwen3_8_27b.matrix_recipes import (
    MatrixSource, MatrixPart, RowRange, Fp8WeightRecipe, Nvfp4WeightRecipe,
)


class Words:
    def __init__(self, values):
        self.values = values

    def get(self, name):
        return self.values[name]


def fp32_word(word):
    return torch.tensor(word, dtype=torch.int32).view(torch.float32)


def test_inventory_has_exact_nvidia_allocation_and_complete_ownership():
    specs = {s.name: s for s in inventory.TENSOR_SPECS}
    assert (inventory.MODEL_ID, inventory.WEIGHTS_ID) == ("qwen3.8-27b", "nvfp4-modelopt")
    assert specs["text/token_embedding"].format == "BF16"
    assert specs["text/token_embedding"].layout == "contiguous-le-v1"
    assert specs["text/draft_head"].format == specs["text/output_head"].format == "NVFP4_F32M"
    for layer in range(64):
        for role in ("gate_up", "down"):
            assert specs[f"text/layers/{layer}/mlp/{role}"].format == "NVFP4_F32M"
    assert len(recipe.FP8_WEIGHT_RECIPES) == 128
    assert len(recipe.NVFP4_WEIGHT_RECIPES) == 129
    owned = ({"text/token_embedding", "text/draft_head", "text/draft_head_token_ids"}
             | set(recipe.FP8_WEIGHTS_BY_NAME) | set(recipe.NVFP4_WEIGHTS_BY_NAME)
             | set(recipe.INPUT_SCALES_BY_NAME) | set(recipe.DENSE_BY_NAME))
    assert owned == set(specs)
    assert len(specs) == 1264
    assert all(encoded_size(s.layout, s.format, s.shape) > 0 for s in specs.values())


def test_fp8_attention_fusion_preserves_all_rows_and_unequal_fp32_scales():
    original = recipe.FP8_WEIGHTS_BY_NAME["text/layers/3/attention/query_key_gate_value"]
    sources = {p.source.name: replace(p.source, shape=(p.source.shape[0], 4))
               for p in original.parts}
    r = replace(original, shape=(14336, 4),
                parts=tuple(replace(p, source=sources[p.source.name]) for p in original.parts))
    values = {}
    for i, source in enumerate(sources.values()):
        codes = ((torch.arange(source.shape[0])[:, None] * 31 +
                  torch.arange(4)[None, :] * 17 + i * 13) % 127).to(torch.uint8)
        values[source.field("weight")] = codes.view(torch.float8_e4m3fn)
        values[source.field("weight_scale")] = fp32_word(0x3d000001 + i * 12345)
    codes, scales = recipe.materialize_fp8_weight(r, Words(values))
    q, key, value = list(sources.values())
    qrows = [h * 512 + j for h in range(24) for j in range(256)]
    grows = [h * 512 + 256 + j for h in range(24) for j in range(256)]
    order = ((q, qrows), (key, list(range(1024))), (q, grows), (value, list(range(1024))))
    cursor = 0
    for source, rows in order:
        count = len(rows)
        assert torch.equal(codes[cursor:cursor+count],
                           values[source.field("weight")].view(torch.uint8)[rows])
        assert torch.equal(scales[cursor:cursor+count].view(torch.int32),
                           values[source.field("weight_scale")].view(torch.int32).expand(count))
        cursor += count
    assert cursor == 14336


def nv_source(name, n=128, k=64):
    s = MatrixSource(name, (n, k))
    codes = ((torch.arange(n)[:, None] * 43 + torch.arange(k//2)[None, :] * 73) % 256).to(torch.uint8)
    scales = ((torch.arange(n)[:, None] * 11 + torch.arange(k//16)[None, :] * 31) % 127).to(torch.uint8)
    return s, {s.field("weight"): codes,
               s.field("weight_scale"): scales.view(torch.float8_e4m3fn),
               s.field("weight_scale_2"): fp32_word(0x3c654321),
               s.field("input_scale"): fp32_word(0x3e123457)}


def test_nvfp4_gate_up_and_draft_are_exact_packed_row_selection(monkeypatch):
    gate, values = nv_source("gate")
    up, more = nv_source("up")
    more[up.field("weight")] ^= 0x91
    values.update(more)
    r = Nvfp4WeightRecipe("fused", (256, 64),
                         tuple(MatrixPart(s, (RowRange(0, 128),)) for s in (gate, up)),
                         (gate, up))
    codes, scales, multiplier = recipe.materialize_nvfp4_weight(r, Words(values))
    assert multiplier == struct.pack("<I", 0x3c654321)
    for i, s in enumerate((gate, up)):
        assert torch.equal(codes[i*128:(i+1)*128], values[s.field("weight")])
        assert torch.equal(scales[i*128:(i+1)*128],
                           values[s.field("weight_scale")].view(torch.uint8))
    head, head_values = nv_source("lm_head", 256)
    head_recipe = Nvfp4WeightRecipe("text/output_head", (256, 64),
                                   (MatrixPart(head, (RowRange(0, 256),)),), (head,))
    monkeypatch.setitem(recipe.NVFP4_WEIGHTS_BY_NAME, "text/output_head", head_recipe)
    ids = torch.arange(255, 0, -2, dtype=torch.int64)
    packed, scale, global_word = recipe.materialize_draft_head(Words(head_values), ids)
    assert torch.equal(packed, head_values["lm_head.weight"][ids])
    assert torch.equal(scale, head_values["lm_head.weight_scale"].view(torch.uint8)[ids])
    assert global_word == multiplier
    with pytest.raises(ValueError, match="duplicate"):
        recipe.materialize_draft_head(Words(head_values), torch.tensor([1, 1], dtype=torch.int64))
    with pytest.raises(ValueError, match="tokenizer domain"):
        recipe.materialize_draft_head(Words(head_values), torch.tensor([248077], dtype=torch.int64))


def metadata():
    return {n: TensorMetadata(n, "model.safetensors", shape, dtype)
            for n, (shape, dtype) in recipe.SOURCE_REQUIREMENTS.items()}


def test_closed_metadata_admission_and_all_scalar_checks():
    class Source:
        names = tuple(recipe.SOURCE_REQUIREMENTS)

        def metadata(self, names):
            return metadata()

        def get(self, name):
            assert recipe.SOURCE_REQUIREMENTS[name] == ((), "F32")
            return fp32_word(0x3e654321)
    result = recipe.preflight_source(Source())
    assert result.source_dtype_counts == {"BF16": 798, "F8_E4M3": 401, "F32": 802, "U8": 193}
    for mutation in ("missing", "extra", "dtype", "shape"):
        source = metadata()
        key = next(iter(source))
        if mutation == "missing":
            del source[key]
        elif mutation == "extra":
            source["extra"] = source[key]
        elif mutation == "dtype":
            source[key] = replace(source[key], dtype="F32")
        else:
            source[key] = replace(source[key], shape=(1,))
        with pytest.raises(ValueError, match="inventory mismatch|signature"):
            recipe.validate_metadata(source)


@pytest.mark.parametrize("field", ["input_scale", "weight_scale_2"])
def test_preflight_rejects_fused_scale_mismatch_before_encoding(field):
    bad = "model.language_model.layers.0.mlp.up_proj." + field
    class Source:
        names = tuple(recipe.SOURCE_REQUIREMENTS)

        def metadata(self, names):
            return metadata()

        def get(self, name):
            return fp32_word(0x3e654322 if name == bad else 0x3e654321)
    with pytest.raises(ValueError, match=f"fused {field} words differ"):
        recipe.preflight_source(Source())


def test_embedding_stream_preserves_original_bf16_words():
    words = torch.tensor([0, -32768, 1, 0x3f81, 0x7fc1, -1] * 7, dtype=torch.int16).reshape(7, 6)
    reader = Words({recipe.EMBEDDING_SOURCE.name: words.view(torch.bfloat16)})
    chunks = list(recipe.iter_embedding_payload(reader, (7, 6), rows_per_chunk=3))
    assert list(map(len, chunks)) == [36, 36, 12]
    assert b"".join(chunks) == words.numpy().tobytes()


def test_modelopt_quantization_config_is_closed(monkeypatch):
    # Model dimensions have their existing family tests; isolate source allocation here.
    monkeypatch.setattr(converter.family_config, "validate_config", lambda c: {})
    q = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
         "ignore": ["mtp*", "mtp.layers.0*"], "producer": {"name": "modelopt"},
         "quantized_layers": {}, "config_groups": {}}
    for key, sources, bits in (("group_0", recipe.FP8_SOURCES, 8),
                               ("group_1", recipe.NVFP4_SOURCES, 4)):
        words = {"dynamic": False, "num_bits": bits, "type": "float"}
        if bits == 4:
            words["group_size"] = 16
        q["config_groups"][key] = {"targets": [s.name for s in sources],
                                   "weights": dict(words), "input_activations": dict(words)}
        q["quantized_layers"].update({s.name: ({"quant_algo": "FP8"} if bits == 8 else
                                             {"quant_algo": "NVFP4", "group_size": 16})
                                       for s in sources})
    converter.validate_config({"quantization_config": q})
    bad = json.loads(json.dumps(q))
    bad["quantized_layers"]["lm_head"]["group_size"] = 32
    with pytest.raises(ValueError, match="registered allocation"):
        converter.validate_config({"quantization_config": bad})
    bad = json.loads(json.dumps(q))
    bad["config_groups"]["group_1"]["input_activations"]["dynamic"] = True
    with pytest.raises(ValueError, match="dynamic"):
        converter.validate_config({"quantization_config": bad})


def test_wrong_output_name_and_device_fail_before_source_read(tmp_path):
    with pytest.raises(ValueError, match="output basename"):
        converter.convert(tmp_path / "missing", tmp_path / "old.ninfer")
    with pytest.raises(ValueError, match="CPU"):
        converter.convert(tmp_path / "missing", tmp_path / converter.OUTPUT_BASENAME, device="cuda")
    assert list(tmp_path.iterdir()) == []


def test_frontend_adapter_reconstructs_registered_metadata_without_changing_tokens():
    tokens = [{"id": 248044, "content": "<|endoftext|>", "lstrip": False,
               "rstrip": False, "normalized": False, "single_word": False, "special": True},
              {"id": 248046, "content": "<|im_end|>", "lstrip": False,
               "rstrip": False, "normalized": False, "single_word": False, "special": True}]
    c = {"add_prefix_space": False, "bos_token": None,
         "clean_up_tokenization_spaces": False, "eos_token": "<|im_end|>",
         "errors": "replace", "model_max_length": 262144, "pad_token": "<|im_end|>",
         "split_special_tokens": False, "tokenizer_class": "Qwen2Tokenizer",
         "unk_token": None, "pretokenize_regex": "literal-regex",
         "model_specific_special_tokens": {"image_token": "<|image_pad|>"}}
    g = {"eos_token_id": [248046, 248044], "top_k": 20, "transformers_version": "5.13.1"}
    before = json.dumps((tokens, c, g))
    a, b = frontend.canonical_metadata({"added_tokens": tokens}, c, "literal-template", g)
    actual = json.loads(a)
    assert actual["add_bos_token"] is False
    assert actual["pad_token"] == "<|endoftext|>"
    assert actual["chat_template"] == "literal-template"
    assert actual["extra_special_tokens"] == c["model_specific_special_tokens"]
    assert actual["added_tokens_decoder"] == {
        str(x["id"]): {k: v for k, v in x.items() if k != "id"} for x in tokens}
    assert json.loads(b) == {"eos_token_id": [248046, 248044], "top_k": 20}
    assert before == json.dumps((tokens, c, g))  # No mutation of source metadata.


def test_shortlist_uses_canonical_specials_even_when_source_config_has_no_decoder(tmp_path, monkeypatch):
    import numpy as np
    monkeypatch.setattr(frontend.draft_head, "VOCAB_SIZE", 16)
    monkeypatch.setattr(frontend.draft_head, "TOKENIZER_VOCAB_SIZE", 12)
    monkeypatch.setattr(frontend.draft_head, "DRAFT_HEAD_N", 8)
    monkeypatch.setattr(frontend.draft_head, "load_total_counts",
                        lambda p: np.arange(16, dtype=np.int64))
    canonical = json.dumps({"added_tokens_decoder": {
        "1": {"special": True}, "3": {"special": True}, "2": {"special": False}}}).encode()
    resources = (converter.family_conversion.ResourcePayload("frontend/tokenizer_config.json", canonical),)
    shortlist = frontend.compute_shortlist(tmp_path / "ranking", tmp_path, resources)
    assert shortlist.force_include == (1, 3)
    assert shortlist.selected.tolist() == [11, 10, 9, 8, 7, 6, 3, 1]
