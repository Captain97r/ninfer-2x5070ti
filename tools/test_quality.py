"""Score fixed inference tasks and compare exact observable responses.

The server must already be running. Run baseline and candidate panels from fresh
server processes with identical model, precision, sampler and runtime settings.
No generated code is executed. This bounded panel is a regression screen.
A capture exits nonzero when any task fails. Comparison can succeed against a
complete reference containing task failures, but only with exact observable
parity and no new task failures; those inherited failures remain explicit.
Supply --runtime-config to bind the comparison to recorded artifact/runtime
settings. The runner records this object; it cannot verify server launch flags.
"""

import argparse
import base64
import copy
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "data" / "quality-panel.json"


def observable(response):
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("expected exactly one response choice")
    choice = choices[0]
    message = choice["message"]
    tools = []
    for call in message.get("tool_calls") or []:
        tools.append({"type": call.get("type"), "function": call["function"]})
    return {
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "reasoning": message.get("reasoning"),
        "tool_calls": tools,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": response["usage"]["completion_tokens"],
    }


def validate_prompt_count(case, actual):
    """Keep prompt metadata outside output parity, but enforce a fixture's exact context."""
    if actual is not None and (type(actual) is not int or actual < 0):
        raise ValueError("usage.prompt_tokens must be a nonnegative integer")
    if "expected_prompt_tokens" not in case:
        return
    expected = case["expected_prompt_tokens"]
    if type(expected) is not int or expected < 0:
        raise ValueError("expected_prompt_tokens must be a nonnegative integer")
    if type(actual) is not int:
        raise ValueError("expected_prompt_tokens requires usage.prompt_tokens")
    if actual != expected:
        raise ValueError(f"prompt token count differs: expected {expected}, received {actual}")


def json_equal(actual, expected):
    # Python considers True == 1. JSON task scoring must distinguish those types.
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            json_equal(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            json_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def strict_json_loads(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonstandard JSON constant: " + value)

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number: " + value)
        return result

    return json.loads(text, object_pairs_hook=object_pairs,
                      parse_constant=invalid_constant, parse_float=finite_float)


def score(expected, answer):
    kind = expected["kind"]
    content = answer["content"] or ""
    if answer["finish_reason"] not in ("stop", "tool_calls"):
        return False, "answer did not finish normally"
    if kind == "tool_call":
        calls = answer["tool_calls"]
        if answer["finish_reason"] != "tool_calls" or content.strip() or len(calls) != 1:
            return False, "expected one tool call and no visible prose"
        call = calls[0]
        if call["type"] != "function" or call["function"]["name"] != expected["name"]:
            return False, "wrong tool"
        try:
            actual = strict_json_loads(call["function"]["arguments"])
        except (ValueError, TypeError):
            return False, "tool arguments are not JSON"
        return json_equal(actual, expected["arguments"]), "tool argument comparison"
    if answer["tool_calls"] or answer["finish_reason"] != "stop":
        return False, "unexpected tool call"
    if kind == "exact_text":
        return content.strip() == expected["value"], "visible text comparison"
    if kind == "json":
        try:
            actual = strict_json_loads(content)
        except (ValueError, TypeError):
            return False, "visible answer is not a complete JSON value"
        return json_equal(actual, expected["value"]), "JSON value comparison"
    raise ValueError(f"unsupported scoring kind: {kind}")


def request_body(case, model, image_bytes):
    body = copy.deepcopy(case["request"])
    body["model"] = model
    body["stream"] = False
    if "image_fixture" in case:
        encoded = base64.b64encode(image_bytes[case["image_fixture"]]).decode("ascii")
        body["messages"].append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
            {"type": "text", "text": case["question"]},
        ]})
    return body


def comparison_contract(raw, cases, model, context, vision, image_bytes, runtime_config):
    return {
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "case_ids": [case["id"] for case in cases],
        "image_fixture_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in image_bytes.items()
        },
        "model": model,
        "context": context,
        "vision": vision,
        "runtime_config": runtime_config,
    }


def evaluate_case(case, answer, previous=None):
    rule_passed, detail = score(case["expected"], answer)
    equal = json_equal(answer, previous["observable"]) if previous is not None else None
    return {
        "id": case["id"], "category": case["category"], "observable": answer,
        "rule_passed": rule_passed, "rule_detail": detail,
        "task_status": "passed" if rule_passed else "failed",
        "baseline_equal": equal,
        "new_task_failure": previous["rule_passed"] and not rule_passed
                            if previous is not None else None,
        "regression_status": ("unchanged" if equal else "changed")
                             if previous is not None else "not_compared",
    }


def panel_summary(rows, case_ids, comparing):
    complete = (
        bool(case_ids) and [row["id"] for row in rows] == case_ids
        and all("error" not in row and "observable" in row
                and type(row.get("rule_passed")) is bool
                and row.get("task_status") in ("passed", "failed") for row in rows)
    )
    all_tasks_passed = complete and all(row["rule_passed"] for row in rows)
    all_equal = complete and all(row["baseline_equal"] is True for row in rows) if comparing else None
    no_new_failures = (
        complete and all(row["new_task_failure"] is False for row in rows)
        if comparing else None
    )
    return {
        "complete": complete,
        "all_tasks_passed": all_tasks_passed,
        "all_observables_equal": all_equal,
        "no_new_task_failures": no_new_failures,
        "regression_passed": all_equal and no_new_failures if comparing else None,
    }


def report_exit_code(report):
    criterion = "regression_passed" if report["mode"] == "comparison" else "all_tasks_passed"
    return 0 if report["complete"] and report[criterion] is True else 1


def validate_reference(reference, contract, cases):
    """Accept complete captures, including genuine task failures, never partial runs."""
    if reference.get("schema_version") != 2:
        raise ValueError("reference schema differs; capture a new reference")
    if not json_equal(reference.get("contract"), contract):
        raise ValueError("reference fixture/image/model/context/runtime contract differs")
    rows = reference.get("cases")
    if (reference.get("complete") is not True or not isinstance(rows, list)
            or len(rows) != len(cases)):
        raise ValueError("reference is incomplete or contains request/scoring errors")
    expected_fields = {
        "content", "reasoning_content", "reasoning", "tool_calls",
        "finish_reason", "completion_tokens",
    }
    for case, row in zip(cases, rows):
        if (not isinstance(row, dict) or row.get("id") != case["id"]
                or "error" in row or not isinstance(row.get("observable"), dict)
                or row["observable"].keys() != expected_fields):
            raise ValueError("reference is incomplete or contains request/scoring errors")
        try:
            passed, _ = score(case["expected"], row["observable"])
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise ValueError("reference has an unscorable response") from error
        validate_prompt_count(case, row.get("prompt_tokens"))
        if (row.get("rule_passed") is not passed
                or row.get("task_status") != ("passed" if passed else "failed")):
            raise ValueError("reference task result does not match its recorded response")
    if reference.get("all_tasks_passed") is not all(row["rule_passed"] for row in rows):
        raise ValueError("reference task summary does not match its recorded responses")
    return {row["id"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:19080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--context", type=int, default=102400)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--baseline", type=Path,
                        help="complete comparison reference; task failures remain failures")
    parser.add_argument("--runtime-config", type=Path,
                        help="JSON object recording artifact and runtime settings for exact comparison")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", default="unspecified")
    args = parser.parse_args()

    raw = args.fixtures.read_bytes()
    fixtures = strict_json_loads(raw)
    cases = [c for c in fixtures["cases"] if args.vision or "image_fixture" not in c]
    ids = [c["id"] for c in cases]
    if not ids or any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("fixture IDs must be nonempty and unique")
    runtime_config = strict_json_loads(args.runtime_config.read_bytes()) if args.runtime_config else None
    if args.runtime_config and not isinstance(runtime_config, dict):
        raise ValueError("--runtime-config must contain a JSON object")
    # Hash exactly the bytes that will be sent, including repeated image fixtures.
    image_bytes = {
        name: (ROOT / "examples" / "cli" / "media" / name).read_bytes()
        for name in {case["image_fixture"] for case in cases if "image_fixture" in case}
    }
    contract = comparison_contract(raw, cases, args.model, args.context, args.vision,
                                   image_bytes, runtime_config)
    prior = {}
    if args.baseline:
        reference = strict_json_loads(args.baseline.read_bytes())
        prior = validate_reference(reference, contract, cases)
    comparing = args.baseline is not None

    url = args.url.rstrip("/")
    with urlopen(url + "/v1/models", timeout=10) as response:
        models = strict_json_loads(response.read())["data"]
    model = next(m for m in models if m["id"] == args.model)
    if model.get("context_length") != args.context:
        raise ValueError("server context does not match the panel configuration")
    if args.vision and "image" not in model.get("input_modalities", []):
        raise ValueError("the image panel requires a vision-enabled server")

    results = []
    for case in cases:
        begin = time.perf_counter()
        row = {"id": case["id"], "category": case["category"], "rule_passed": None,
               "task_status": "error", "baseline_equal": None, "prompt_tokens": None,
               "new_task_failure": None, "regression_status": "error" if comparing else "not_compared"}
        try:
            body = request_body(case, args.model, image_bytes)
            request = Request(url + "/v1/chat/completions",
                              data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=180) as response:
                decoded = strict_json_loads(response.read())
            row["prompt_tokens"] = decoded["usage"].get("prompt_tokens")
            validate_prompt_count(case, row["prompt_tokens"])
            answer = observable(decoded)
            row.update(evaluate_case(case, answer, prior.get(case["id"])))
        except HTTPError as error:
            row["error"] = f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')[:2000]}"
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as error:
            row["error"] = str(error)
        row["wall_seconds"] = time.perf_counter() - begin
        results.append(row)
        print(f"{case['id']}: task={row['task_status']} "
              f"regression={row['regression_status']}", flush=True)

    report = {
        "schema_version": 2, "mode": "comparison" if comparing else "capture",
        "label": args.label, "date_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "scope": "Bounded task scoring and exact observable-response parity; not token-ID or broad quality equivalence.",
        **panel_summary(results, ids, comparing), "cases": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
                           + "\n", encoding="utf-8")
    print(f"complete={report['complete']} all_tasks_passed={report['all_tasks_passed']} "
          f"all_observables_equal={report['all_observables_equal']} "
          f"regression_passed={report['regression_passed']}", flush=True)
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
