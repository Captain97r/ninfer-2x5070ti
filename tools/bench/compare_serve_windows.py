"""Compare nine fixed, short-prompt serving requests using an owned Windows server.

Outputs and performance are reported separately; this is not task-quality scoring.
The unchanged corpus prompts deliberately request more than the 1024-token cap.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.bench import run_serve_corpus as corpus
from tools.test_quality import json_equal, observable


FIXTURES = ("scenario_code_python", "scenario_story_en_mystery", "scenario_structured_jsonl")
# Frozen subset of the existing corpus seeds, selected before observing outputs.
SEEDS = (7632647173703958409, 7968175640111700217, 912910298659544128)
MAX_COMPLETION_TOKENS = 1024
HOST, PORT = "127.0.0.1", 19080
ARTIFACT_TYPE = "ninfer_tp2_serving_comparison"
SCHEMA_VERSION = 1
RUNTIME_OPTIONS = [
    "--tp", "2", "--devices", "0,1", "--max-context", "102400",
    "--kv-capacity", "102400", "--kv-dtype", "int8", "--prefill-chunk", "1024",
    "--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft",
    "--vision", "--image-max-tokens", "2048", "--max-concurrency", "1",
    "--default-max-tokens", "102400", "--log-stats-interval-ms", "0",
]
EXPECTED_ENGINE = {
    "tp": 2, "devices": [0, 1], "max_context": 102400,
    # MemorySummary reports the native rotary admission ceiling here, separately
    # from the configured session limit above and resolved KV capacity below.
    "effective_max_context": 262144,
    "kv_capacity": 102400, "prefill_chunk": 1024, "kv_cache": "int8-group64",
    "vision": True, "max_concurrency": 1, "cuda_graph": True, "prefix_reuse": True,
    "speculative_backend": "mtp", "speculative_draft_window": 3, "proposal_head": "optimized",
}
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def specs(weights):
    result = []
    for name in FIXTURES:
        path = ROOT / "examples/cli/messages" / (name + ".json")
        fixture = corpus.Fixture(name, json.loads(path.read_text(encoding="utf-8")),
                                 False, MAX_COMPLETION_TOKENS, "scenario",
                                 corpus.fixture_metadata(name)[1])
        for seed in SEEDS:
            result.append(corpus.RunSpec("qwen3_8_27b", "qwen3.8-27b", weights,
                                         "mtp3", "mtp", 3, "stochastic", fixture, seed))
    return result


def validate_start(event):
    corpus.require_server_log_identity(event, "server_start")
    engine = event.get("engine", {})
    if {key: engine.get(key) for key in EXPECTED_ENGINE} != EXPECTED_ENGINE:
        raise corpus.CampaignError("server configuration differs from the fixed TP2 workload")
    server = event.get("server", {})
    if (server.get("host"), server.get("port"), server.get("public_model_id")) != (
            HOST, PORT, "qwen3.8-27b"):
        raise corpus.CampaignError("unexpected server endpoint or model")
    artifact = event.get("artifact", {})
    if (artifact.get("target"), artifact.get("weights_id")) != ("qwen3_8_27b", "nvfp4"):
        raise corpus.CampaignError("expected the registered Qwen3.8 NVFP4 artifact")
    defaults = event.get("sampling_defaults", {})
    preset = defaults.get("non_thinking", {})
    overrides = defaults.get("server_overrides", {})
    if (defaults.get("greedy") is not False or set(preset) != set(SAMPLING_FIELDS)
            or set(overrides) != {*SAMPLING_FIELDS, "seed"}
            or any(value is not None for value in overrides.values())
            or not isinstance(preset.get("temperature"), (int, float))
            or preset["temperature"] <= 0):
        raise corpus.CampaignError("expected unchanged registered nonthinking stochastic defaults")
    if not isinstance(event.get("server_instance_id"), str) or not event["server_instance_id"]:
        raise corpus.CampaignError("server_start has no instance identity")
    return preset


def nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise corpus.CampaignError(name + " must be a nonnegative integer")
    return value


def checked_record(spec, payload, response, event, preset):
    """Reuse corpus metrics, but account for enabled prefix reuse and reject bad data."""
    result = event.get("result", {})
    usage = response.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens"):
        nonnegative_int(result.get(key), "result." + key)
        nonnegative_int(usage.get(key), "usage." + key)
    computed = nonnegative_int(result.get("computed_prefill_tokens"), "computed_prefill_tokens")
    reused = nonnegative_int(result.get("prefix_cache_hit_tokens"), "prefix_cache_hit_tokens")
    if computed + reused != result["prompt_tokens"]:
        raise corpus.CampaignError("computed and reused prompt counts do not add up")
    if not 0 < result["completion_tokens"] <= MAX_COMPLETION_TOKENS:
        raise corpus.CampaignError("completion count is empty or exceeds the fixed cap")
    timings = event.get("timings_seconds", {})
    for key in ("prepare", "vision", "prefill", "decode", "total", "ttft"):
        value = timings.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise corpus.CampaignError("invalid timing: " + key)
    if ((computed and timings["prefill"] <= 0)
            or (result["completion_tokens"] > 1 and timings["decode"] <= 0)):
        raise corpus.CampaignError("positive token work has no measured duration")
    speculative = event.get("speculative", {})
    for key in ("rounds", "drafted_tokens", "accepted_tokens", "fallback_steps"):
        nonnegative_int(speculative.get(key), "speculative." + key)
    if speculative.get("draft_window") != 3 or speculative["accepted_tokens"] > speculative["drafted_tokens"]:
        raise corpus.CampaignError("invalid MTP counters or draft window")
    positions = speculative.get("accepted_per_position")
    if not isinstance(positions, list):
        raise corpus.CampaignError("missing accepted-per-position counts")
    for count in positions:
        nonnegative_int(count, "accepted_per_position")
    if sum(positions) != speculative["accepted_tokens"]:
        raise corpus.CampaignError("accepted-per-position counts do not add up")
    actual_sampling = event.get("request", {}).get("sampling", {})
    if actual_sampling != {**preset, "seed": spec.seed}:
        raise corpus.CampaignError("request did not use registered defaults and its fixed seed")
    answer = observable(response)
    # The log retains the Engine reason; OpenAI groups the two limits as length
    # and the two normal stop causes as stop. Tool responses have their own finish.
    server_finish = result.get("finish_reason")
    mapped_finish = {"output_limit": "length", "context_capacity": "length",
                     "stop_token": "stop", "stop_string": "stop"}.get(server_finish)
    tool_count = nonnegative_int(result.get("tool_call_count"), "tool_call_count")
    if mapped_finish is None or tool_count != len(answer["tool_calls"]):
        raise corpus.CampaignError("unexpected Engine finish reason or tool count")
    if answer["finish_reason"] != ("tool_calls" if tool_count else mapped_finish):
        raise corpus.CampaignError("HTTP and server log finish reasons differ")
    record = corpus.build_result_record(spec, "nvfp4", payload, response, event)
    # The corpus disables prefix reuse; its total-prompt numerator would inflate
    # PP throughput here. A full cache hit has no meaningful PP rate.
    record["metrics"].update({
        "computed_prefill_tokens": computed, "prefix_cache_hit_tokens": reused,
        "prefix_reuse_path": result.get("prefix_reuse_path"),
        "prefill_tok_s": corpus.safe_ratio(computed, timings["prefill"]) if computed else None,
        "accepted_per_position": positions,
        "server_ttft_ms": 1000.0 * timings["ttft"],
        "server_finish_reason": server_finish, "finish_reason": answer["finish_reason"],
    })
    record["observable"] = answer
    return record


def summarize(records):
    summary = []
    for name in FIXTURES:
        selected = [record for record in records if record["fixture"] == name]
        row = {"fixture": name, "requests": len(selected),
               "finish_reasons": [record["metrics"]["finish_reason"] for record in selected]}
        for metric in ("prompt_tokens", "computed_prefill_tokens", "prefix_cache_hit_tokens",
                       "completion_tokens", "prefill_tok_s", "decode_tok_s", "server_ttft_ms",
                       "speculative_acceptance", "speculative_tokens_per_round"):
            n, mean, stddev = corpus.sample_stats(selected, metric)
            row[metric] = {"n": n, "mean": mean, "sample_stddev": stddev}
        summary.append(row)
    return summary


def compare_reports(current, baseline):
    if (baseline.get("artifact_type"), baseline.get("schema_version")) != (ARTIFACT_TYPE, SCHEMA_VERSION):
        raise corpus.CampaignError("baseline is not a serving comparison report")
    if not json_equal(current["contract"], baseline.get("contract")):
        raise corpus.CampaignError("baseline artifact, runtime, requests, or warmup contract differs")
    previous = baseline.get("requests", [])
    records = current["requests"]
    if len(previous) != len(records) or len(records) != len(FIXTURES) * len(SEEDS):
        raise corpus.CampaignError("baseline does not contain the complete fixed request set")
    comparisons = []
    for record, old in zip(records, previous):
        if (record["fixture"], record["seed"]) != (old.get("fixture"), old.get("seed")) or not json_equal(
                record["request"], old.get("request")):
            raise corpus.CampaignError("baseline request identity or order differs")
        answer, old_answer = observable(record["response"]), observable(old["response"])
        changed = [key for key in answer if not json_equal(answer[key], old_answer.get(key))]
        metrics, old_metrics = record["metrics"], old["metrics"]
        same_prompt = metrics["prompt_tokens"] == old_metrics["prompt_tokens"]
        comparisons.append({
            "fixture": record["fixture"], "seed": record["seed"],
            "observable_equal": not changed, "changed_observable_fields": changed,
            "prompt_tokens_equal": same_prompt,
            "prefix_counts_equal": all(metrics[key] == old_metrics[key] for key in (
                "computed_prefill_tokens", "prefix_cache_hit_tokens")),
            "decode_tok_s_ratio": corpus.safe_ratio(metrics["decode_tok_s"], old_metrics["decode_tok_s"])
                if metrics["decode_tok_s"] is not None and old_metrics["decode_tok_s"] is not None else None,
            "speculative_counts_equal": all(metrics[key] == old_metrics[key] for key in (
                "speculative_rounds", "drafted_tokens", "accepted_tokens", "fallback_steps", "accepted_per_position")),
        })
    return {
        "baseline_label": baseline.get("label"),
        "all_observables_equal": all(row["observable_equal"] for row in comparisons),
        "all_prompt_counts_equal": all(row["prompt_tokens_equal"] for row in comparisons),
        "task_quality_scored": False, "requests": comparisons,
    }


def assert_port_free():
    with socket.socket() as check:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            check.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        check.bind((HOST, PORT))


def execute(command, output, all_specs, report, baseline):
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PATH"] = str(ROOT / "build/windows/vcpkg_installed/x64-windows/bin") + os.pathsep + env["PATH"]
    log_path = output / "requests.jsonl"
    server = corpus.RunningServer(command, HOST, PORT, log_path)
    # Only this Popen instance is terminated, including exceptions and interrupts.
    # The fresh directory owns both files; no old JSONL data can be mistaken for readiness.
    with (output / "server.log").open("xb") as console:
        assert_port_free()
        server.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console,
                                          stderr=subprocess.STDOUT,
                                          creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            server.tail = corpus.ServerLogTail(log_path, server.process, 0)
            start = server.wait_until_ready()
            preset = validate_start(start)
            report["server_start"] = start
            report["contract"]["resolved_nonthinking_sampling"] = preset
            if baseline and not json_equal(report["contract"], baseline.get("contract")):
                raise corpus.CampaignError("baseline contract differs before inference")
            warmup_specs = [all_specs[index * len(SEEDS)] for index in range(len(FIXTURES))]
            connection = http.client.HTTPConnection(HOST, PORT, timeout=180)
            last_id = None
            try:
                with (output / "records.jsonl").open("x", encoding="utf-8") as records_file:
                    for phase, sequence in (("warmup", warmup_specs), ("requests", all_specs)):
                        for spec in sequence:
                            payload = corpus.request_payload(spec.model_id, spec.fixture, spec.seed)
                            before = time.monotonic()
                            response = corpus.post_json(connection, payload)
                            wall_seconds = time.monotonic() - before
                            event = server.wait_for_request_done(start["server_instance_id"])
                            request_id = nonnegative_int(event.get("request", {}).get("request_id"), "request_id")
                            if last_id is not None and request_id != last_id + 1:
                                raise corpus.CampaignError("unexpected concurrent or missing serving request")
                            last_id = request_id
                            record = checked_record(spec, payload, response, event, preset)
                            record.update({"phase": phase, "http_wall_seconds": wall_seconds})
                            corpus.append_record(records_file, record)
                            report[phase].append(record)
                            print(f"{phase}: {spec.fixture.name} seed={spec.seed}, "
                                  f"tokens={record['metrics']['completion_tokens']}, "
                                  f"finish={record['metrics']['finish_reason']}", flush=True)
            finally:
                connection.close()
        finally:
            server.stop()
            print("Owned comparison server stopped.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline", type=Path, help="report.json from the reference binary")
    parser.add_argument("--weights", type=Path, help="defaults to config/model.json's artifact path")
    args = parser.parse_args(argv)
    if os.name != "nt" or sys.version_info[:2] != (3, 11):
        raise corpus.CampaignError("use Python 3.11 on Windows")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise corpus.CampaignError("label must contain only letters, numbers, underscores and hyphens")
    model = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    weights = (args.weights or Path(model["default_directory"]) / model["filename"]).resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    if baseline is not None:
        # Reject incomplete/wrong-format baselines before loading the model.
        compare_reports(baseline, baseline)
    all_specs = specs(weights)
    assert_port_free()
    output = ROOT / "build/serving-comparison" / args.label
    output.mkdir(parents=True, exist_ok=False)
    command = [str(binary), str(weights), *RUNTIME_OPTIONS, "--host", HOST, "--port", str(PORT),
               "--request-log-jsonl", str(output / "requests.jsonl")]
    print("Fingerprinting the artifact and binary; fixed workload will be recorded before launch.", flush=True)
    contract = {
        "artifact_sha256": sha256(weights), "artifact_bytes": weights.stat().st_size,
        "runtime_options": RUNTIME_OPTIONS,
        "requests": [corpus.request_payload(spec.model_id, spec.fixture, spec.seed) for spec in all_specs],
        "warmup_request_indices": [0, 3, 6],
        "request_order": "fixture-major, seed-minor; all warmups precede all measurements",
        "prefix_reuse": True,
    }
    report = {
        "artifact_type": ARTIFACT_TYPE, "schema_version": SCHEMA_VERSION, "label": args.label,
        "contract": contract,
        "launch": {"command": command, "binary_sha256": sha256(binary)},
        "scope": "Nine short-prompt scenario requests, three seeds each, capped at 1024 completion tokens. "
                 "Three same-cap warmups exercise representative decode graphs; early EOS is retained. "
                 "Prefix reuse stays enabled. PP uses computed tokens; TG uses completion_tokens minus one. "
                 "Truncated outputs are retained; exact parity is not a task-quality score. "
                 "Separate process runs and three seeds do not establish a broad performance claim.",
        "warmup": [], "requests": [],
    }
    write_json(output / "workload.json", report)
    try:
        execute(command, output, all_specs, report, baseline)
        report["summary"] = summarize(report["requests"])
        report["comparison"] = compare_reports(report, baseline) if baseline else None
        write_json(output / "report.json", report)
    except BaseException as error:
        report["error"] = str(error)
        write_json(output / "failed.json", report)
        raise
    print("Report: " + str(output / "report.json"), flush=True)
    for row in report["summary"]:
        tg = row["decode_tok_s"]
        if tg["mean"] is not None:
            print(f"{row['fixture']}: TG {tg['mean']:.2f} +/- {tg['sample_stddev']:.2f} tok/s "
                  f"across {row['requests']} seeds; finishes={row['finish_reasons']}", flush=True)
    comparison = report["comparison"]
    if comparison:
        matched = sum(item["observable_equal"] for item in comparison["requests"])
        print(f"Exact observable matches: {matched}/9. No task-quality score.", flush=True)
    return 0 if comparison is None or (comparison["all_observables_equal"] and comparison["all_prompt_counts_equal"]) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (corpus.CampaignError, OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        raise SystemExit(2) from None
