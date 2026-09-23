"""Exact source words and independently indexed physical planes for ModelOpt formats."""
from __future__ import annotations

import math
import struct

import pytest
import torch

from tools.artifact.container import Artifact, ArtifactIdentity, TensorSpec, write_artifact
from tools.artifact.layouts import (
    encode_nvfp4, decode_nvfp4_words, encode_fp8_row_scaled,
    decode_fp8_row_scaled_words, dequantize_fp8_row_scaled, encoded_size,
)
from tools.artifact.numeric import decode_e2m1_word, decode_e4m3fn_word

NV = "NVFP4_F32M"
FP = "FP8_E4M3FN_ROW_F32S"
NV_LAYOUT = "blockscale-k16-m128x4-multiplier-v1"
FP_LAYOUT = "row-scale-f32-v1"


def e4(word):
    sign = -1.0 if word & 128 else 1.0
    exponent, fraction = (word >> 3) & 15, word & 7
    return sign * (fraction / 512.0 if exponent == 0 else
                   (8 + fraction) * 2.0 ** (exponent - 10))


def test_nvfp4_multiplier_planes_and_complete_formula(tmp_path):
    n, k = 256, 128
    packed = ((torch.arange(n * k // 2).reshape(n, k // 2) * 73 + 19) % 256).to(torch.uint8)
    scales = ((torch.arange(n * k // 16).reshape(n, k // 16) * 47 + 5) % 127).to(torch.uint8)
    multiplier_bytes = struct.pack("<I", 0x3eaaaaab)  # FP32 1/3; never store its reciprocal.
    multiplier = struct.unpack("<f", multiplier_bytes)[0]
    payload = encode_nvfp4(packed, scales, multiplier_bytes, (n, k), format=NV)
    code_bytes, scale_bytes = n * k // 2, n * k // 16
    assert len(payload) == code_bytes + scale_bytes + 4
    assert payload[:code_bytes] == packed.numpy().tobytes()
    assert payload[-4:] == multiplier_bytes
    decoded, natural, global_scale = decode_nvfp4_words(payload, (n, k), format=NV)
    assert torch.equal(decoded, packed)
    assert torch.equal(natural, scales)
    assert global_scale.view(torch.int32).item() == 0x3eaaaaab
    magnitudes = (0.0, .5, 1., 1.5, 2., 3., 4., 6.)
    # Independent scalar indexing covers both M128 blocks and both K64 tiles.
    for row in range(n):
        for col in range(k):
            group = col // 16
            offset = (((row // 128 * (k // 64) + group // 4) * 32
                       + row % 32) * 4 + row % 128 // 32) * 4 + group % 4
            scale_word = payload[code_bytes + offset]
            assert scale_word == int(scales[row, group])
            nibble = (payload[row * (k // 2) + col // 2] >> (4 * (col % 2))) & 15
            value = math.copysign(magnitudes[nibble & 7], -1. if nibble & 8 else 1.)
            expected = value * e4(scale_word) * multiplier
            actual = (decode_e2m1_word(nibble) *
                      decode_e4m3fn_word(int(natural[row, group])) * float(global_scale))
            assert actual == expected
            assert math.copysign(1., actual) == math.copysign(1., expected)
    path = tmp_path / "preserved.ninfer"
    spec = TensorSpec("weight", (n, k), NV, NV_LAYOUT)
    write_artifact(path, ArtifactIdentity("test", "modelopt"), [(spec, payload)])
    with Artifact.open(path) as artifact:
        assert artifact.objects[0].format == NV
        assert artifact.objects[0].layout == NV_LAYOUT
        assert bytes(artifact.payload("weight")) == payload


def test_fp8_f32_scale_words_full_formula_and_padding(tmp_path):
    words = [x for x in range(256) if (x & 127) != 127]
    codes = torch.tensor([[x, words[(i + 61) % 254], words[(i + 119) % 254]]
                          for i, x in enumerate(words)], dtype=torch.uint8)
    scale_words = torch.tensor([0x3d000001 + i * 8191 for i in range(254)], dtype=torch.int32)
    scales = scale_words.view(torch.float32)
    shape = tuple(codes.shape)
    payload = encode_fp8_row_scaled(codes, scales, shape, format=FP)
    scale_offset = (codes.numel() + 255) // 256 * 256
    assert payload[:codes.numel()] == codes.numpy().tobytes()
    assert payload[codes.numel():scale_offset] == bytes(scale_offset - codes.numel())
    assert payload[scale_offset:] == scale_words.numpy().tobytes()
    actual_codes, actual_scales = decode_fp8_row_scaled_words(payload, shape, format=FP)
    assert torch.equal(actual_codes, codes)
    assert torch.equal(actual_scales.view(torch.int32), scale_words)
    values = dequantize_fp8_row_scaled(payload, shape, torch.float64, format=FP)
    for r in range(shape[0]):
        for c in range(shape[1]):
            expected = e4(int(codes[r, c])) * float(scales[r])
            assert float(values[r, c]) == expected
            assert math.copysign(1., float(values[r, c])) == math.copysign(1., expected)
    spec = TensorSpec("weight", shape, FP, FP_LAYOUT)
    path = tmp_path / "rows.ninfer"
    write_artifact(path, ArtifactIdentity("test", "modelopt"), [(spec, payload)])
    with Artifact.open(path) as artifact:
        assert bytes(artifact.payload("weight")) == payload
        assert artifact.objects[0].bytes == encoded_size(FP_LAYOUT, FP, shape)


@pytest.mark.parametrize("bad", [0x80000000, 0xbf800000, 0x7f800000, 0x7fc00001])
def test_fp32_row_scale_rejects_noncanonical_words(bad):
    signed = bad if bad < 2**31 else bad - 2**32
    scales = torch.tensor([signed], dtype=torch.int32).view(torch.float32)
    with pytest.raises(ValueError, match="nonnegative finite FP32"):
        encode_fp8_row_scaled(torch.zeros((1, 3), dtype=torch.uint8), scales, (1, 3), format=FP)


def test_new_formats_require_exact_typed_scales_and_distinct_layouts():
    with pytest.raises(TypeError, match="FP32"):
        encode_fp8_row_scaled(torch.zeros((1, 3), dtype=torch.uint8),
                             torch.ones(1, dtype=torch.bfloat16), (1, 3), format=FP)
    with pytest.raises(ValueError, match="zero row scale"):
        encode_fp8_row_scaled(torch.full((1, 3), 0x38, dtype=torch.uint8),
                             torch.zeros(1, dtype=torch.float32), (1, 3), format=FP)
    for layout, format, shape in (
        ("row-scale-v1", FP, (1, 3)),
        ("blockscale-k16-m128x4-v1", NV, (128, 64)),
        (FP_LAYOUT, "FP8_E4M3FN_ROW_BF16S", (1, 3)),
        (NV_LAYOUT, "NVFP4", (128, 64)),
    ):
        with pytest.raises(ValueError, match="does not accept"):
            encoded_size(layout, format, shape)


def test_inherited_bf16_scale_reconstruction_keeps_binary32_boundary():
    codes = torch.tensor([[0x7e]], dtype=torch.uint8)  # E4M3FN maximum, 448.
    scales = torch.tensor([0x7f7f], dtype=torch.int16).view(torch.bfloat16)
    old = encode_fp8_row_scaled(codes, scales, (1, 1))
    assert torch.isinf(dequantize_fp8_row_scaled(old, (1, 1), torch.float64)).all()
    new = encode_fp8_row_scaled(codes, scales.float(), (1, 1), format=FP)
    exact = dequantize_fp8_row_scaled(new, (1, 1), torch.float64, format=FP)
    assert exact.item() == 448.0 * scales.double().item()
    assert torch.isfinite(exact).all()
