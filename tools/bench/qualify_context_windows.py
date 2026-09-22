"""Qualify near-limit retrieval, context exhaustion, and post-session health.

Four fixed requests run in one owned Windows server. Exact JSON task scoring,
context-boundary exercise, and baseline observable parity are separate outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tools.bench import context_sweep_windows as sweep
from tools.bench import compare_serve_windows as comparison
from tools.bench import run_serve_corpus as corpus
from tools.test_quality import json_equal, observable, score, validate_prompt_count

SEED = 42
GENERATOR = "ninfer-near-limit-ledger-v1"
CASE_NAMES = ("retrieve-three", "absent-key", "context-boundary", "post-session-health")


def digest(case, field, row):
    return hashlib.sha256(f"{GENERATOR}|{case}|{field}|{row}".encode("ascii")).hexdigest().upper()


def greedy_payload(model, system, user, cap):
    return {"model": model, "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}],
            "temperature": 0, "seed": SEED, "max_completion_tokens": cap,
            "enable_thinking": False, "stream": False}


def ledger_case(model, name, rows):
    if name not in CASE_NAMES[:2] or rows < 100:
        raise ValueError("ledger requires its fixed case identity and at least 100 records")
    found = name == "retrieve-three"
    queries = ["REC-" + digest(name, "query", index)[:16] for index in range(3 if found else 1)]
    values = ["VAL-" + digest(name, "answer", index)[:20] for index in range(len(queries))]
    locations = [int((rows - 1) * depth / 100) for depth in (5, 50, 95)] if found else []
    replacement = {location: index for index, location in enumerate(locations)}
    records, keys = [], set()
    for row in range(rows):
        key = "REC-" + digest(name, "key", row)[:16]
        value = "VAL-" + digest(name, "value", row)[:20]
        if key in queries:
            raise ValueError("generated distractor collides with a query")
        if row in replacement:
            index = replacement[row]
            key, value = queries[index], values[index]
        if key in keys:
            raise ValueError("generated ledger keys are not unique")
        keys.add(key)
        units = 1 + int(digest(name, "units", row)[:6], 16) % 997
        records.append(f"Record {row:06d}: key={key}; value={value}; units={units}.")
    if any((key in keys) != found for key in queries):
        raise ValueError("ledger query presence differs from its oracle")
    system = (f"Case identity: {name}. Use only the supplied ledger. Return exactly one JSON object "
              "mapping each requested key to its exact recorded value. If a key is absent, use JSON null. "
              "Never infer values from neighboring records. Include every requested key and no other keys. "
              "Return no explanation, code fence, or markdown.")
    user = "Ledger begins.\n" + "\n".join(records) + "\nLedger ends.\nRequested keys: " + ", ".join(queries)
    return {"name": name, "kind": "json", "records": rows, "needle_rows": locations,
            "needle_record_depth_percent": [5, 50, 95] if found else [],
            "generator": GENERATOR,
            "expected": {"kind": "json", "value": dict(zip(queries, values if found else [None]))},
            "payload": greedy_payload(model, system, user, 256)}


def calibrate_ledger(model, name, context, counter=sweep.count_payload):
    target, low, high, memo = context - 512, 100, max(100, (context - 512) // 40), {}
    def measure(rows):
        if rows not in memo:
            case = ledger_case(model, name, rows)
            memo[rows] = counter(case["payload"])
        return memo[rows]
    if measure(low) > target:
        raise corpus.CampaignError("context is too small for the fixed ledger minimum")
    while measure(high) <= target:
        low, high = high, high * 2
    while high - low > 1:
        middle = (high + low) // 2
        if measure(middle) <= target:
            low = middle
        else:
            high = middle
    count = measure(low)
    if not target - 128 <= count <= target:
        raise corpus.CampaignError("ledger could not fill the near-limit budget within 128 tokens")
    case = ledger_case(model, name, low)
    return {**case, "target_prompt_tokens": target, "expected_prompt_tokens": count,
            "count_endpoint_calls": len(memo)}


def settings(context):
    return {"context": context, "runtime_options": sweep.runtime_options(context),
            "ledger_target": context - 512, "ledger_completion_cap": 256,
            "boundary_target": context - 2048, "boundary_completion_cap": 8192,
            "greedy_temperature": 0, "seed": SEED, "case_order": list(CASE_NAMES)}


def load_workload(path, context):
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if (frozen.get("artifact_type") != "ninfer_context_qualification_workload"
            or frozen.get("schema_version") != 1 or not json_equal(frozen.get("settings"), settings(context))):
        raise corpus.CampaignError("replay context/runtime/fixture contract differs")
    cases = frozen.get("cases", [])
    if [case.get("name") for case in cases] != list(CASE_NAMES):
        raise corpus.CampaignError("replay case set or order differs")
    for index, case in enumerate(cases):
        payload = case.get("payload", {})
        cap = 256 if index < 2 else 8192 if index == 2 else 128
        # Replay the complete saved text, never regenerate against a changed checkout.
        if (set(payload) != {"model", "messages", "temperature", "seed", "max_completion_tokens",
                             "enable_thinking", "stream"}
                or payload.get("model") != "qwen3.8-27b" or payload.get("temperature") != 0
                or payload.get("seed") != SEED or payload.get("max_completion_tokens") != cap
                or payload.get("enable_thinking") is not False or payload.get("stream") is not False):
            raise corpus.CampaignError("replay generation controls differ")
        count = comparison.nonnegative_int(case.get("expected_prompt_tokens"), "expected_prompt_tokens")
        if index < 3:
            target, tolerance = (context - 512, 128) if index < 2 else (context - 2048, sweep.TOKEN_TOLERANCE)
            if case.get("target_prompt_tokens") != target or not target - tolerance <= count <= target:
                raise corpus.CampaignError("replay prompt budget differs")
        if case.get("kind") != ("boundary" if index == 2 else "json"):
            raise corpus.CampaignError("replay scoring contract differs")
    return frozen


def build_workload(model, context, output):
    cases = [calibrate_ledger(model, name, context) for name in CASE_NAMES[:2]]
    body, manifest = sweep.freeze_sources("docs", output)
    boundary = sweep.calibrate(model, "docs", body, manifest, context - 2048, 8192)
    boundary["payload"].update(temperature=0, seed=SEED)
    # Sampling controls do not change the template; verify that fact with the final payload.
    if sweep.count_payload(boundary["payload"]) != boundary["expected_prompt_tokens"]:
        raise corpus.CampaignError("boundary final request changed its prompt count")
    cases.append({**boundary, "name": "context-boundary", "kind": "boundary"})
    health = {"name": "post-session-health", "kind": "json",
              "expected": {"kind": "json", "value": {"status": "ok", "check": 7}},
              "payload": greedy_payload(model, "Follow the user's output format exactly.",
                                         'Return exactly this JSON object: {"status":"ok","check":7}', 128)}
    health["expected_prompt_tokens"] = sweep.count_payload(health["payload"])
    cases.append(health)
    return {"artifact_type": "ninfer_context_qualification_workload", "schema_version": 1,
            "settings": settings(context), "cases": cases}


def boundary_outcome(prompt, completion, context, engine_finish):
    # request_plan_impl.h reserves P+L-1 positions: the final emitted token need
    # not be evaluated into KV. A full context therefore emits P+L=context+1.
    used = prompt + completion - 1
    result = {"cached_token_frontier": used, "prompt_plus_completion": prompt + completion,
              "expected_cached_frontier": context,
              "contract": "prompt_tokens + completion_tokens -1 == context at context_capacity"}
    if engine_finish == "context_capacity":
        result["status"] = "passed" if used == context else "failed"
    elif engine_finish in ("stop_token", "stop_string") and used < context:
        result["status"] = "not_exercised"
    else:
        result["status"] = "failed"
    return result


def assess(case, response, event, context, preset, weights):
    payload = case["payload"]
    result, usage = event.get("result", {}), response.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens"):
        comparison.nonnegative_int(result.get(key), "result." + key)
        comparison.nonnegative_int(usage.get(key), "usage." + key)
    prompt, completion = result["prompt_tokens"], result["completion_tokens"]
    validate_prompt_count(case, usage.get("prompt_tokens"))
    if (prompt != case["expected_prompt_tokens"] or not 0 < completion <= payload["max_completion_tokens"]
            or result.get("computed_prefill_tokens") != prompt or result.get("prefix_cache_hit_tokens") != 0):
        raise corpus.CampaignError("count, completion budget, or cold-prefill contract failed")
    if event.get("request", {}).get("sampling") != {**preset, "temperature": 0, "seed": SEED}:
        raise corpus.CampaignError("request was not greedy with its fixed seed/default penalties")
    answer = observable(response)
    engine_finish = result.get("finish_reason")
    mapped = {"context_capacity": "length", "output_limit": "length", "stop_token": "stop",
              "stop_string": "stop"}.get(engine_finish)
    if (mapped is None or answer["finish_reason"] != mapped
            or result.get("tool_call_count") != 0 or answer["tool_calls"]):
        raise corpus.CampaignError("HTTP and engine finish/tool metadata differ")
    fixture = corpus.Fixture(case["name"], payload["messages"], False, payload["max_completion_tokens"],
                             "context-qualification", case["kind"])
    spec = corpus.RunSpec("qwen3_8_27b", payload["model"], weights, "mtp3", "mtp", 3, "greedy", fixture, SEED)
    record = corpus.build_result_record(spec, "nvfp4", payload, response, event)
    record["observable"] = answer
    record["metrics"].update(server_finish_reason=engine_finish, finish_reason=mapped,
                              server_ttft_ms=1000 * event["timings_seconds"]["ttft"])
    record["validated"] = True
    if case["kind"] == "boundary":
        record["boundary"] = boundary_outcome(prompt, completion, context, engine_finish)
        record["task_passed"] = None
    else:
        passed, detail = score(case["expected"], answer)
        record.update(task_passed=passed, task_score_detail=detail)
    return record


def summarize(records):
    complete = len(records) == 4 and all(record.get("response") is not None for record in records)
    validated = complete and all(record.get("validated") is True for record in records)
    tasks = [record for record in records if record.get("fixture") != "context-boundary"]
    all_tasks = validated and len(tasks) == 3 and all(record.get("task_passed") is True for record in tasks)
    boundary = next((record.get("boundary", {}).get("status", "failed") for record in records
                     if record.get("fixture") == "context-boundary"), "not_run")
    return {"complete": complete, "all_observations_validated": validated,
            "all_tasks_passed": all_tasks, "boundary_status": boundary,
            "qualification_passed": all_tasks and boundary == "passed"}


def compare_reference(report, path):
    reference = json.loads(path.read_text(encoding="utf-8"))
    if (reference.get("artifact_type") != "ninfer_context_qualification"
            or reference.get("schema_version") != 1 or reference.get("summary", {}).get("complete") is not True
            or reference.get("summary", {}).get("all_observations_validated") is not True
            or not json_equal(reference.get("workload"), report["workload"])
            or not json_equal(reference.get("sampling"), report["sampling"])):
        raise corpus.CampaignError("baseline is incomplete or has a different workload/sampler")
    pairs = []
    old_records = reference.get("records", [])
    if len(old_records) != 4 or len(report["records"]) != 4:
        raise corpus.CampaignError("baseline/current observation set is incomplete")
    if any(old.get("validated") is not True or "error" in old for old in old_records):
        raise corpus.CampaignError("baseline contains unvalidated or errored observations")
    for current, old in zip(report["records"], old_records):
        if current["fixture"] != old.get("fixture"):
            raise corpus.CampaignError("baseline observation order differs")
        same = json_equal(observable(current["response"]), observable(old["response"]))
        pairs.append({"fixture": current["fixture"], "observable_equal": same,
                      "prompt_count_equal": current["response"]["usage"]["prompt_tokens"] == old["response"]["usage"]["prompt_tokens"],
                      "new_task_failure": old.get("task_passed") is True and current.get("task_passed") is not True})
    return {"reference_label": reference.get("label"), "cases": pairs,
            "all_observables_equal": all(row["observable_equal"] and row["prompt_count_equal"] for row in pairs),
            "no_new_task_failures": not any(row["new_task_failure"] for row in pairs)}


def execute(args, binary, weights, output, report):
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PATH"] = str(ROOT / "build/windows/vcpkg_installed/x64-windows/bin") + os.pathsep + env["PATH"]
    command = [str(binary), str(weights), *sweep.runtime_options(args.context), "--host", sweep.HOST,
               "--port", str(sweep.PORT), "--request-log-jsonl", str(output / "requests.jsonl")]
    report["command"] = command
    server = corpus.RunningServer(command, sweep.HOST, sweep.PORT, output / "requests.jsonl")
    with (output / "server.log").open("xb") as console:
        comparison.assert_port_free()
        server.process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=console,
                                          stderr=subprocess.STDOUT, creationflags=sweep.CREATE_NO_WINDOW)
        try:
            server.tail = corpus.ServerLogTail(server.log_path, server.process, 0)
            start = server.wait_until_ready()
            preset = sweep.validate_start(start, args.context)
            report.update(server_start=start, sampling=preset)
            models = sweep.read_json("/v1/models").get("data", [])
            if len(models) != 1 or models[0].get("id") != "qwen3.8-27b":
                raise corpus.CampaignError("unexpected model discovery")
            if args.replay_from:
                frozen = load_workload(args.replay_from, args.context)
                if not json_equal(frozen.get("sampling"), preset):
                    raise corpus.CampaignError("replay registered sampling defaults differ")
                for case in frozen["cases"]:
                    if sweep.count_payload(case["payload"]) != case["expected_prompt_tokens"]:
                        raise corpus.CampaignError("replay tokenizer/template changed the prompt")
            else:
                frozen = build_workload(models[0]["id"], args.context, output)
                frozen["sampling"] = preset
            comparison.write_json(output / "workload.json", frozen)
            report["workload"] = frozen
            with (output / "records.jsonl").open("x", encoding="utf-8") as handle:
                for index, case in enumerate(frozen["cases"], 1):
                    print(f"{case['name']}: {case['expected_prompt_tokens']} prompt tokens", flush=True)
                    before, response, event = time.monotonic(), None, None
                    try:
                        response = sweep.read_json("/v1/chat/completions", case["payload"])
                        _, event = sweep.correlate(server, start["server_instance_id"], index)
                        record = assess(case, response, event, args.context, preset, weights)
                    except (Exception, KeyboardInterrupt) as error:
                        record = {"fixture": case["name"], "request": case["payload"], "response": response,
                                  "server_event": event, "validated": False, "error": str(error),
                                  "http_status": getattr(error, "status", None),
                                  "raw_error_body": getattr(error, "body", None)}
                        if isinstance(error, KeyboardInterrupt):
                            corpus.append_record(handle, record)
                            report["records"].append(record)
                            raise
                    record["http_wall_seconds"] = time.monotonic() - before
                    corpus.append_record(handle, record)
                    report["records"].append(record)
                    print(f"  validated={record['validated']} task={record.get('task_passed')} "
                          f"boundary={record.get('boundary', {}).get('status')}", flush=True)
        finally:
            server.stop()
            print("Owned qualification server stopped.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--replay-from", type=Path, help="baseline workload.json; sibling report.json is compared if present")
    args = parser.parse_args(argv)
    if os.name != "nt" or sys.version_info[:2] != (3, 11):
        raise corpus.CampaignError("use Python3.11 on Windows")
    if not 16384 <= args.context <= 262144 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise corpus.CampaignError("context must be16384..262144 and label must be filesystem-safe")
    if args.replay_from:
        args.replay_from = args.replay_from.resolve(strict=True)
        load_workload(args.replay_from, args.context)
    model = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    weights = (args.weights or Path(model["default_directory"]) / model["filename"]).resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    comparison.assert_port_free()
    output = ROOT / "build/context-qualification" / args.label
    output.mkdir(parents=True, exist_ok=False)
    report = {"artifact_type": "ninfer_context_qualification", "schema_version": 1,
              "label": args.label, "context": args.context, "binary_sha256": comparison.sha256(binary),
              "replay_from": str(args.replay_from) if args.replay_from else None, "records": [],
              "scope": "Two deterministic synthetic ledger tasks, unique real-document boundary request, "
                       "then short JSON health request. Greedy, cold, no forced continuation. Task failure "
                       "does not abort remaining observations. Early EOS does not qualify context exhaustion. "
                       "At context_capacity, prompt+completion-1 equals the configured context; the last "
                       "emitted token does not require a KV position. Exact parity is a separate outcome."}
    try:
        execute(args, binary, weights, output, report)
        report["summary"] = summarize(report["records"])
        reference = args.replay_from.parent / "report.json" if args.replay_from else None
        if reference and reference.exists():
            report["comparison"] = compare_reference(report, reference)
        comparison.write_json(output / "report.json", report)
    except BaseException as error:
        report["error"] = str(error)
        report["summary"] = summarize(report["records"])
        comparison.write_json(output / "failed.json", report)
        raise
    print(json.dumps(report["summary"]), flush=True)
    print("Report: " + str(output / "report.json"), flush=True)
    parity = report.get("comparison", {})
    return 0 if (report["summary"]["qualification_passed"]
                 and parity.get("all_observables_equal", True)
                 and parity.get("no_new_task_failures", True)) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (corpus.CampaignError, OSError, ValueError) as error:
        print("error: " + str(error), file=sys.stderr)
        raise SystemExit(2) from None
