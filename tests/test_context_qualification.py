"""CPU checks for exact near-limit oracles, limit accounting, and report outcomes."""

import copy
import json
from pathlib import Path
import re
import tempfile
import unittest

from tools.bench import qualify_context_windows as qualify
from tools.bench import run_serve_corpus as corpus


class ContextQualificationTests(unittest.TestCase):
    def test_records_preserve_both_admitted_server_profiles(self):
        preset = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                  "presence_penalty": 1.5, "frequency_penalty": 0.0}
        case = {"name": "post-session-health", "kind": "json", "expected_prompt_tokens": 100,
                "expected": {"kind": "json", "value": {"status": "ok"}},
                "payload": qualify.greedy_payload("qwen3.8-27b", "instruction", "question", 128)}
        response = {"choices": [{"message": {"content": '{"status":"ok"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
        event = {"artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
                 "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION, "event": "request_done",
                 "request": {"model": "qwen3.8-27b", "requested_output_tokens": 128,
                             "enable_thinking": False,
                             "sampling": {**preset, "temperature": 0, "seed": qualify.SEED}},
                 "result": {"prompt_tokens": 100, "completion_tokens": 5, "computed_prefill_tokens": 100,
                            "prefix_cache_hit_tokens": 0, "finish_reason": "stop_token", "tool_call_count": 0},
                 "timings_seconds": {"prepare": 0.1, "vision": 0, "prefill": 2, "decode": 1,
                                     "total": 3.1, "ttft": 2.2},
                 "speculative": {"backend": "mtp", "rounds": 2, "drafted_tokens": 6,
                                 "accepted_tokens": 2, "fallback_steps": 0}}
        for weights_id in ("nvfp4", "nvfp4-modelopt"):
            start = {"artifact": {"target": "qwen3_8_27b", "weights_id": weights_id}}
            qualify.comparison.validate_artifact_identity(start)
            with self.subTest(weights_id=weights_id):
                record = qualify.assess(case, response, event, 102400, preset, Path("model.ninfer"),
                                        weights_id=start["artifact"]["weights_id"])
                self.assertEqual(record["weights_id"], weights_id)
                self.assertTrue(record["validated"])
                self.assertTrue(record["task_passed"])

    def test_ledger_oracle_matches_three_distinct_record_depths_and_absent_control(self):
        for name in ("retrieve-three", "absent-key"):
            case = qualify.ledger_case("qwen3.8-27b", name, 401)
            body = case["payload"]["messages"][1]["content"]
            rows = re.findall(r"Record (\d+): key=(REC-[A-F0-9]+); value=(VAL-[A-F0-9]+); units=(\d+)\.", body)
            self.assertEqual(len(rows), 401)
            actual = {key: value for _, key, value, _ in rows}
            self.assertEqual(len(actual), 401)
            if name == "retrieve-three":
                self.assertEqual(case["needle_rows"], [20, 200, 380])
                self.assertEqual(len(set(case["expected"]["value"].values())), 3)
                for index, (key, value) in zip(case["needle_rows"], case["expected"]["value"].items()):
                    self.assertEqual(rows[index][1:3], (key, value))
            else:
                self.assertEqual(case["needle_rows"], [])
                for key, value in case["expected"]["value"].items():
                    self.assertNotIn(key, actual)
                    self.assertIsNone(value)
            observed = {key: actual.get(key) for key in case["expected"]["value"]}
            self.assertEqual(observed, case["expected"]["value"])
        # Different complete record counts change positions, not queried values.
        a = qualify.ledger_case("qwen3.8-27b", "retrieve-three", 401)
        b = qualify.ledger_case("qwen3.8-27b", "retrieve-three", 799)
        self.assertEqual(a["expected"], b["expected"])
        self.assertNotEqual(a["needle_rows"], b["needle_rows"])

    def test_record_calibration_uses_complete_unique_records_with_independent_count(self):
        def count(payload):
            body = payload["messages"][1]["content"]
            return 200 + 60 * len(re.findall(r"Record \d+:", body))
        case = qualify.calibrate_ledger("qwen3.8-27b", "retrieve-three", 16384, count)
        self.assertLessEqual(case["expected_prompt_tokens"], 16384 - 512)
        self.assertGreaterEqual(case["expected_prompt_tokens"], 16384 - 512 - 128)
        self.assertEqual(count(case["payload"]), case["expected_prompt_tokens"])

    def test_context_limit_counts_final_emitted_token_without_allocating_its_kv(self):
        result = qualify.boundary_outcome(194560, 2049, 196608, "context_capacity")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["cached_token_frontier"], 196608)
        self.assertEqual(result["prompt_plus_completion"], 196609)
        self.assertEqual(qualify.boundary_outcome(194560, 2048, 196608, "context_capacity")["status"], "failed")
        self.assertEqual(qualify.boundary_outcome(194560, 800, 196608, "stop_token")["status"], "not_exercised")
        self.assertEqual(qualify.boundary_outcome(194560, 2049, 196608, "output_limit")["status"], "failed")

    def test_complete_task_success_and_exercised_boundary_are_separate(self):
        records = [{"fixture": name, "response": {}, "validated": True, "task_passed": True}
                   for name in qualify.CASE_NAMES]
        records[2].update(task_passed=None, boundary={"status": "not_exercised"})
        result = qualify.summarize(records)
        self.assertTrue(result["complete"])
        self.assertTrue(result["all_tasks_passed"])
        self.assertFalse(result["qualification_passed"])
        records[2]["boundary"]["status"] = "passed"
        self.assertTrue(qualify.summarize(records)["qualification_passed"])
        records[0]["task_passed"] = False
        failed_task = qualify.summarize(records)
        self.assertTrue(failed_task["complete"])
        self.assertFalse(failed_task["all_tasks_passed"])
        records[0].update(response=None, validated=False)
        self.assertFalse(qualify.summarize(records)["complete"])

    def test_reference_requires_valid_observations_but_allows_task_failures(self):
        response = {"choices": [{"message": {"content": "wrong but valid answer"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
        records = [{"fixture": name, "response": copy.deepcopy(response), "validated": True,
                    "task_passed": False} for name in qualify.CASE_NAMES]
        report = {"artifact_type": "ninfer_context_qualification", "schema_version": 1,
                  "label": "baseline", "workload": {"frozen": True}, "sampling": {"temperature": 0},
                  "summary": {"complete": True, "all_observations_validated": True,
                              "all_tasks_passed": False}, "records": records}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(qualify.compare_reference(report, path)["all_observables_equal"])
            mutations = [lambda r: r["summary"].update(all_observations_validated=False),
                         lambda r: r["records"][0].update(validated=False),
                         lambda r: r["records"][0].update(error="count mismatch")]
            for mutate in mutations:
                invalid = copy.deepcopy(report)
                mutate(invalid)
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(mutation=mutate), self.assertRaises(corpus.CampaignError):
                    qualify.compare_reference(report, path)

    def test_replay_rejects_changed_budgets_seed_or_request_order(self):
        context = 196608
        cases = []
        for index, name in enumerate(qualify.CASE_NAMES):
            cap = 256 if index < 2 else 8192 if index == 2 else 128
            target = context - 512 if index < 2 else context - 2048 if index == 2 else 64
            cases.append({"name": name, "kind": "boundary" if index == 2 else "json",
                          "target_prompt_tokens": target, "expected_prompt_tokens": target,
                          "payload": qualify.greedy_payload("qwen3.8-27b", "instruction", "frozen old body", cap)})
        frozen = {"artifact_type": "ninfer_context_qualification_workload", "schema_version": 1,
                  "settings": qualify.settings(context), "cases": cases}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            self.assertEqual(qualify.load_workload(path, context), frozen)
            mutations = [lambda f: f["cases"].reverse(),
                         lambda f: f["cases"][0]["payload"].update(seed=43),
                         lambda f: f["cases"][0].update(expected_prompt_tokens=context - 1000),
                         lambda f: f["settings"].update(context=102400)]
            for mutate in mutations:
                changed = copy.deepcopy(frozen)
                mutate(changed)
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(mutation=mutate), self.assertRaises(corpus.CampaignError):
                    qualify.load_workload(path, context)


if __name__ == "__main__":
    unittest.main()
