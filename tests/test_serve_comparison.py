"""CPU checks for serving comparison metric attribution and false parity."""

import copy
import math
import unittest
from pathlib import Path

from tools.bench import compare_serve_windows as comparison
from tools.bench import run_serve_corpus as corpus


PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
          "presence_penalty": 1.5, "frequency_penalty": 0.0}


def response_and_event(spec):
    response = {"id": "random-id", "choices": [{
        "message": {"content": "first line\n", "reasoning_content": None},
        "finish_reason": "length"}], "usage": {"prompt_tokens": 100, "completion_tokens": 11}}
    event = {
        "artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
        "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION, "event": "request_done",
        "request": {"model": spec.model_id, "requested_output_tokens": 1024,
                    "enable_thinking": False, "sampling": {**PRESET, "seed": spec.seed}},
        "result": {"prompt_tokens": 100, "completion_tokens": 11, "finish_reason": "output_limit", "tool_call_count": 0,
                   "computed_prefill_tokens": 20, "prefix_cache_hit_tokens": 80,
                   "prefix_reuse_path": "append"},
        "timings_seconds": {"prepare": 0.01, "vision": 0, "prefill": 0.25,
                            "decode": 0.5, "total": 0.76, "ttft": 0.3},
        "speculative": {"backend": "mtp", "draft_window": 3, "rounds": 4,
                        "drafted_tokens": 12, "accepted_tokens": 6, "fallback_steps": 0,
                        "accepted_per_position": [3, 2, 1]},
    }
    return response, event


def record(spec):
    response, event = response_and_event(spec)
    payload = corpus.request_payload(spec.model_id, spec.fixture, spec.seed)
    return comparison.checked_record(spec, payload, response, event, PRESET)


def report():
    return {"artifact_type": comparison.ARTIFACT_TYPE,
            "schema_version": comparison.SCHEMA_VERSION, "label": "reference",
            "contract": {"requests": ["fixed"], "runtime": ["same"]},
            "requests": [record(spec) for spec in comparison.specs(Path("model.ninfer"))]}


class ServingComparisonTests(unittest.TestCase):
    def setUp(self):
        self.spec = comparison.specs(Path("model.ninfer"))[0]
        self.payload = corpus.request_payload(self.spec.model_id, self.spec.fixture, self.spec.seed)

    def checked(self, response, event):
        return comparison.checked_record(self.spec, self.payload, response, event, PRESET)

    def test_native_ceiling_is_distinct_from_configured_session_and_kv(self):
        # Relevant fields from the reproduced startup: the native RoPE ceiling
        # is 262144 even when this session and its KV allocation are both 102400.
        event = {
            "artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
            "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION, "event": "server_start",
            "server_instance_id": "owned-test-instance",
            "engine": {
                "tp": 2, "devices": [0, 1], "max_context": 102400,
                "effective_max_context": 262144, "kv_capacity": 102400,
                "prefill_chunk": 1024, "kv_cache": "int8-group64", "vision": True,
                "max_concurrency": 1, "cuda_graph": True, "prefix_reuse": True,
                "speculative_backend": "mtp", "speculative_draft_window": 3,
                "proposal_head": "optimized", "rope_mode": "native",
            },
            "server": {"host": "127.0.0.1", "port": 19080, "public_model_id": "qwen3.8-27b"},
            "artifact": {"target": "qwen3_8_27b", "weights_id": "nvfp4"},
            "sampling_defaults": {"greedy": False, "non_thinking": PRESET,
                                  "server_overrides": {key: None for key in (*PRESET, "seed")}},
        }
        self.assertEqual(comparison.validate_start(event), PRESET)
        for field, changed in (("max_context", 262144), ("kv_capacity", 262144),
                               ("effective_max_context", 102400)):
            invalid = copy.deepcopy(event)
            invalid["engine"][field] = changed
            with self.subTest(field=field), self.assertRaises(corpus.CampaignError):
                comparison.validate_start(invalid)

    def test_prefix_work_and_first_token_accounting(self):
        response, event = response_and_event(self.spec)
        metrics = self.checked(response, event)["metrics"]
        self.assertEqual(metrics["prefill_tok_s"], 80)  # 20 computed, not 100 total.
        self.assertEqual(metrics["decode_tokens"], 10)
        self.assertEqual(metrics["decode_tok_s"], 20)
        self.assertEqual(metrics["speculative_acceptance"], 0.5)
        self.assertEqual(metrics["server_ttft_ms"], 300)  # Authoritative first token; not 260ms phase sum.
        self.assertEqual(metrics["finish_reason"], "length")
        self.assertEqual(metrics["server_finish_reason"], "output_limit")
        self.assertEqual(metrics["accepted_per_position"], [3, 2, 1])
        event["result"].update(computed_prefill_tokens=0, prefix_cache_hit_tokens=100)
        self.assertIsNone(self.checked(response, event)["metrics"]["prefill_tok_s"])

    def test_engine_finish_mapping_and_tool_count_are_preserved(self):
        for engine, wire in (("output_limit", "length"), ("context_capacity", "length"),
                             ("stop_token", "stop"), ("stop_string", "stop")):
            response, event = response_and_event(self.spec)
            response["choices"][0]["finish_reason"] = wire
            event["result"]["finish_reason"] = engine
            self.assertEqual(self.checked(response, event)["metrics"]["server_finish_reason"], engine)
        response, event = response_and_event(self.spec)
        event["result"].update(finish_reason="stop_token", tool_call_count=1)
        response["choices"][0].update(finish_reason="tool_calls")
        response["choices"][0]["message"]["tool_calls"] = [
            {"id": "random", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        self.assertEqual(self.checked(response, event)["metrics"]["finish_reason"], "tool_calls")
        event["result"]["tool_call_count"] = 0
        with self.assertRaises(corpus.CampaignError):
            self.checked(response, event)

    def test_rejects_misattributed_or_malformed_metrics(self):
        mutations = [
            lambda r, e: r["usage"].update(completion_tokens=12),
            lambda r, e: r["choices"][0].update(finish_reason="stop"),
            lambda r, e: e["request"]["sampling"].update(seed=self.spec.seed + 1),
            lambda r, e: e["request"]["sampling"].update(temperature=0),
            lambda r, e: e["result"].update(computed_prefill_tokens=21),
            lambda r, e: e["result"].update(completion_tokens=True),
            lambda r, e: e["timings_seconds"].update(decode=math.nan),
            lambda r, e: e["timings_seconds"].update(decode=0),
            lambda r, e: e["speculative"].update(accepted_per_position=[3, 1, 1]),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                response, event = response_and_event(self.spec)
                mutate(response, event)
                with self.assertRaises(corpus.CampaignError):
                    self.checked(response, event)

    def test_parity_is_separate_from_metrics_and_preserves_raw_answer(self):
        baseline = report()
        candidate = copy.deepcopy(baseline)
        candidate["requests"][0]["response"]["id"] = "different-id"
        candidate["requests"][0]["metrics"]["decode_tok_s"] *= 2
        compared = comparison.compare_reports(candidate, baseline)
        self.assertTrue(compared["all_observables_equal"])
        self.assertFalse(compared["task_quality_scored"])
        self.assertEqual(compared["requests"][0]["decode_tok_s_ratio"], 2)
        # A capped length finish can match exactly without being a completed task.
        mutations = [
            lambda r: r["choices"][0]["message"].update(content="first line"),
            lambda r: r["choices"][0]["message"].update(reasoning_content="reason"),
            lambda r: r["choices"][0].update(finish_reason="stop"),
            lambda r: r["usage"].update(completion_tokens=12),
            lambda r: r["choices"][0]["message"].update(tool_calls=[{
                "id": "call-id", "type": "function", "function": {"name": "f", "arguments": "{} "}}]),
        ]
        for mutate in mutations:
            candidate = copy.deepcopy(baseline)
            mutate(candidate["requests"][0]["response"])
            self.assertFalse(comparison.compare_reports(candidate, baseline)["all_observables_equal"])

    def test_wrong_baseline_contract_order_or_missing_request_is_not_comparable(self):
        current = report()
        for alteration in ("contract", "order", "missing", "payload"):
            baseline = copy.deepcopy(current)
            if alteration == "contract":
                baseline["contract"]["runtime"] = ["different"]
            elif alteration == "order":
                baseline["requests"][0], baseline["requests"][1] = baseline["requests"][1], baseline["requests"][0]
            elif alteration == "missing":
                baseline["requests"].pop()
            else:
                baseline["requests"][0]["request"]["max_completion_tokens"] = 512
            with self.subTest(alteration=alteration), self.assertRaises(corpus.CampaignError):
                comparison.compare_reports(current, baseline)


if __name__ == "__main__":
    unittest.main()
