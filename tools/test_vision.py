"""Run bounded image-grounding checks against an already running NInfer server.

Uses repository fixtures and Python's standard library. Does not start a server.
"""

import argparse
import base64
import json
import re
import struct
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "examples" / "cli" / "media"


def image_part(name):
    encoded = base64.b64encode((MEDIA / name).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}


def message(name, question):
    return {"role": "user", "content": [image_part(name), {"type": "text", "text": question}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--context", type=int, default=102400)
    parser.add_argument("--report", type=Path, default=ROOT / "diagnostics" / "vision-validation.json")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    with urlopen(base + "/v1/models", timeout=10) as response:
        models = json.load(response)
    model = next(item for item in models["data"] if item["id"] == args.model)
    assert model["max_model_len"] == args.context, model
    assert model["input_modalities"] == ["text", "image"], model
    assert model["architecture"]["input_modalities"] == ["text", "image"], model
    with urlopen(base + "/v1/models/" + args.model, timeout=10) as response:
        single = json.load(response)
    for key in ("max_model_len", "context_length", "input_modalities", "architecture"):
        assert single[key] == model[key], key

    cases = []

    def chat(name, messages, stream=False, thinking=False, limit=128):
        body = {"model": args.model, "messages": messages, "temperature": 0,
                "max_completion_tokens": limit, "stream": stream,
                "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": True}}
        if thinking:
            body["chat_template_kwargs"]["reasoning_effort"] = "xhigh"
        if stream:
            body["stream_options"] = {"include_usage": True}
        request = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        begin = time.perf_counter()
        try:
            with urlopen(request, timeout=120) as response:
                payload = response.read().decode()
        except HTTPError as error:
            raise RuntimeError(f"{name}: HTTP {error.code}: {error.read().decode()}") from error
        content, reasoning, finish, usage = "", "", None, None
        if stream:
            lines = payload.splitlines()
            assert "data: [DONE]" in lines, f"{name}: incomplete SSE"
            for line in lines:
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                assert "error" not in event, event
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning_content") or ""
                    finish = choice.get("finish_reason") or finish
        else:
            result = json.loads(payload)
            choice = result["choices"][0]
            content = choice["message"].get("content") or ""
            reasoning = choice["message"].get("reasoning_content") or ""
            finish = choice["finish_reason"]
            usage = result.get("usage")
        assert finish == "stop", f"{name}: unexpected finish={finish}, content={content!r}"
        if thinking:
            assert reasoning, f"{name}: thinking output missing"
        else:
            assert not reasoning, f"{name}: thinking was not disabled"
        record = {"case": name, "stream": stream, "thinking": thinking, "content": content,
                  "finish_reason": finish, "usage": usage,
                  "elapsed_seconds": round(time.perf_counter() - begin, 3)}
        cases.append(record)
        print(json.dumps(record), flush=True)
        return content

    chart_question = (
        'Read the image. Return only JSON with three fields: "heading_number" (the number at '
        'the end of the heading, integer), "red_circles" (the count, integer), and '
        '"blue_square_position" ("left" or "right" of the green triangle).'
    )
    answer = chat("chart_ocr_count_position", [message("visual_chart.png", chart_question)])
    data = json.loads(answer[answer.index("{"):answer.rindex("}") + 1])
    assert data == {"heading_number": 731, "red_circles": 3, "blue_square_position": "left"}, data

    count_question = "How many red circles are in this image? Answer only the integer."
    left = message("compare_left.png", count_question)
    answer = chat("stream_left_image", [left], stream=True, limit=32)
    assert re.fullmatch(r"\s*2[.!]?\s*", answer), answer
    # The right fixture contains three red circles, a blue square and a yellow star.
    right = message("compare_right.png", count_question)
    answer = chat("changed_image_same_question", [right], stream=True, limit=32)
    assert re.fullmatch(r"\s*3[.!]?\s*", answer), answer

    history = [left, {"role": "assistant", "content": "2"}, right]
    answer = chat("two_image_conversation", history, stream=True, limit=32)
    assert re.fullmatch(r"\s*3[.!]?\s*", answer), answer

    answer = chat("thinking_image", [right], stream=True, thinking=True, limit=512)
    assert re.fullmatch(r"\s*3[.!]?\s*", answer), answer
    # A generated solid-color fixture exceeds the default 2 MP image budget.
    # No expected color appears in the question or in the image's metadata.
    def png_chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    # The second image reaches the default 2048-token envelope exactly: 8192 patches,
    # merged 2x2, with no resize. Together these exercise both preprocessing branches.
    for name, width, height in (
        ("oversized_image_downscale", 2560, 1440),
        ("exact_capacity_image", 2048, 1024),
    ):
        scanlines = (b"\x00" + bytes((0, 0, 255)) * width) * height
        png = (b"\x89PNG\r\n\x1a\n"
               + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + png_chunk(b"IDAT", zlib.compress(scanlines)) + png_chunk(b"IEND", b""))
        large = {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": "What is the background color of this image? Answer only the color name."},
        ]}
        answer = chat(name, [large], limit=32)
        assert re.fullmatch(r"\s*blue[.!]?\s*", answer, re.IGNORECASE), answer
        cases[-1]["source_dimensions"] = [width, height]

    answer = chat("text_after_images", [{"role": "user", "content": "Reply with exactly: vision ready"}], limit=16)
    assert "vision ready" in answer.lower(), answer

    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "server": base,
              "model": args.model, "context_capacity": args.context,
              "scope": "HTTP image-grounding integration; not numerical vision-kernel qualification",
              "passed": len(cases), "failed": 0, "cases": cases}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Passed {len(cases)} vision/text checks; report: {args.report}")


if __name__ == "__main__":
    main()
