"""Map pinned ModelOpt export metadata to the registered Qwen3.8 frontend.

Vocabulary, added tokens, template and media metadata come from the source.
Two metadata files are canonicalized; source files are never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import draft_head
from .convert import OFFICIAL_RESOURCE_SHA256
from .inventory import RESOURCE_SPECS

SOURCE_RESOURCE_SHA256 = dict(OFFICIAL_RESOURCE_SHA256)
SOURCE_RESOURCE_SHA256.update({
    "frontend/tokenizer_config.json": "e5d078b00e6c1223b32444db8c1001dc71d86ceef8ee706b5bf084c3a43a1f9c",
    "frontend/generation_config.json": "8a3a817a9d330f6f9e62f048fe62a48594946c4a5c4cb314a8011316d5094986",
})
NORMALIZED_RESOURCES = ("frontend/tokenizer_config.json", "frontend/generation_config.json")


def canonical_metadata(tokenizer: dict, source_config: dict, template: str,
                       generation: dict) -> tuple[bytes, bytes]:
    # These are registered frontend semantics, not inferred ModelOpt defaults.
    decoder = {
        str(v["id"]): {k: v[k] for k in
                      ("content", "lstrip", "normalized", "rstrip", "single_word", "special")}
        for v in tokenizer["added_tokens"]
    }
    c = {
        "add_prefix_space": source_config["add_prefix_space"],
        "added_tokens_decoder": decoder,
        "additional_special_tokens": [
            "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
            "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>",
            "<|video_pad|>",
        ],
        "bos_token": source_config["bos_token"],
        "chat_template": template,
        **{k: source_config[k] for k in (
            "clean_up_tokenization_spaces", "eos_token", "errors", "model_max_length")},
        "pad_token": "<|endoftext|>",
        **{k: source_config[k] for k in ("split_special_tokens", "tokenizer_class", "unk_token")},
        "add_bos_token": False,
        "pretokenize_regex": source_config["pretokenize_regex"],
        "extra_special_tokens": source_config["model_specific_special_tokens"],
    }
    g = {k: v for k, v in generation.items() if k != "transformers_version"}
    encode = lambda x: (json.dumps(x, ensure_ascii=False, indent=4) + "\n").encode("utf-8")
    return encode(c), encode(g)


def load_resources(model_dir: str | Path) -> tuple[family_conversion.ResourcePayload, ...]:
    resources = family_conversion.load_resources(model_dir, RESOURCE_SPECS)
    raw = {r.name: r.data for r in resources}
    for name, expected in SOURCE_RESOURCE_SHA256.items():
        if hashlib.sha256(raw[name]).hexdigest() != expected:
            raise ValueError(f"pinned ModelOpt frontend resource mismatch: {name}")
    config, generation = canonical_metadata(
        json.loads(raw["frontend/tokenizer.json"]),
        json.loads(raw["frontend/tokenizer_config.json"]),
        raw["frontend/chat_template.jinja"].decode("utf-8"),
        json.loads(raw["frontend/generation_config.json"]),
    )
    raw[NORMALIZED_RESOURCES[0]], raw[NORMALIZED_RESOURCES[1]] = config, generation
    for name, expected in OFFICIAL_RESOURCE_SHA256.items():
        if hashlib.sha256(raw[name]).hexdigest() != expected:
            raise ValueError(f"canonical Qwen3.8 frontend reconstruction mismatch: {name}")
    return tuple(family_conversion.ResourcePayload(r.name, raw[r.name]) for r in resources)


def compute_shortlist(ranking: Path, model_dir: Path,
                      resources: tuple[family_conversion.ResourcePayload, ...]):
    config = json.loads(next(r.data for r in resources
                             if r.name == "frontend/tokenizer_config.json"))
    forced = tuple(sorted(int(k) for k, v in config["added_tokens_decoder"].items()
                          if v.get("special", False)))
    counts = draft_head.load_total_counts(ranking)
    selected = draft_head.select_shortlist(
        counts[:draft_head.TOKENIZER_VOCAB_SIZE], draft_head.DRAFT_HEAD_N, forced
    )
    return draft_head.DraftHeadContext(
        n=draft_head.DRAFT_HEAD_N, selected=selected, ranking=ranking,
        tokenizer=model_dir, force_include=forced,
    )
