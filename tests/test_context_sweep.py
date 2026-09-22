"""CPU regression checks for cold-context timing, replay, and memory attribution."""

import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.bench import context_sweep_windows as sweep
from tools.bench import run_serve_corpus as corpus


PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
          "presence_penalty": 1.5, "frequency_penalty": 0.0}


def identity(event, timestamp=0):
    return {"artifact_type": corpus.SERVER_LOG_ARTIFACT_TYPE,
            "schema_version": corpus.SERVER_LOG_SCHEMA_VERSION, "event": event,
            "server_instance_id": "owned", "timestamp_unix_ms": timestamp}


def case_and_result():
    case = {"name": "code-8192-r1", "workload": "code", "target_prompt_tokens": 8192,
            "expected_prompt_tokens": 8180,
            "payload": sweep.make_payload("qwen3.8-27b", "code", "source", 4096)}
    response = {"choices": [{"message": {"content": "Preserve this newline.\n", "reasoning_content": None},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 8180, "completion_tokens": 1001}}
    event = {**identity("request_done", 12000),
             "request": {"request_id": 1, "model": "qwen3.8-27b", "requested_output_tokens": 4096,
                         "enable_thinking": False, "sampling": {**PRESET, "seed": sweep.SEED}},
             "result": {"prompt_tokens": 8180, "completion_tokens": 1001, "computed_prefill_tokens": 8180,
                        "prefix_cache_hit_tokens": 0, "finish_reason": "stop_token", "tool_call_count": 0},
             "timings_seconds": {"prepare": 0.1, "vision": 0, "prefill": 2, "decode": 10,
                                 "total": 12.1, "ttft": 2.4},
             "speculative": {"backend": "mtp", "draft_window": 3, "rounds": 400,
                             "drafted_tokens": 1200, "accepted_tokens": 600, "fallback_steps": 0,
                             "accepted_per_position": [300, 200, 100]}}
    return case, response, event


def interval(end, decode, *, seconds=1, prefill=0, running=1):
    return {**identity("throughput", end), "interval_seconds": seconds,
            "tokens": {"computed_prefill": prefill, "committed_decode": decode},
            "scheduler": {"running": running, "prefilling": int(prefill != 0), "waiting": 0}}


class ContextSweepTests(unittest.TestCase):
    def test_cold_pp_authoritative_ttft_and_early_eos_are_preserved(self):
        case, response, event = case_and_result()
        result = sweep.checked_record(case, Path("model.ninfer"), response, event, PRESET)
        self.assertEqual(result["metrics"]["prefill_tok_s"], 4090)
        self.assertEqual(result["metrics"]["decode_tok_s"], 100)
        self.assertEqual(result["metrics"]["server_ttft_ms"], 2400)  # Not the 2100ms phase sum.
        self.assertTrue(result["metrics"]["early_stop"])
        self.assertFalse(result["metrics"]["reached_cap"])
        self.assertEqual(result["observable"]["content"], "Preserve this newline.\n")
        response["usage"]["completion_tokens"] = event["result"]["completion_tokens"] = 4096
        response["choices"][0]["finish_reason"] = "length"
        event["result"]["finish_reason"] = "output_limit"
        capped = sweep.checked_record(case, Path("model.ninfer"), response, event, PRESET)
        self.assertTrue(capped["metrics"]["reached_cap"])
        self.assertFalse(capped["metrics"]["early_stop"])

    def test_wrong_budget_cache_counts_sampling_and_malformed_usage_fail(self):
        mutations = [
            lambda c, r, e: c.update(expected_prompt_tokens=8190),
            lambda c, r, e: e["result"].update(computed_prefill_tokens=8179, prefix_cache_hit_tokens=1),
            lambda c, r, e: r["usage"].update(completion_tokens=1000),
            lambda c, r, e: r["usage"].update(prompt_tokens=True),
            lambda c, r, e: e["timings_seconds"].update(decode=math.nan),
            lambda c, r, e: e["request"]["sampling"].update(seed=sweep.SEED + 1),
            lambda c, r, e: e["speculative"].update(accepted_per_position=[300, 100, 100]),
            lambda c, r, e: r["choices"][0].update(finish_reason="length"),
        ]
        for mutate in mutations:
            case, response, event = case_and_result()
            mutate(case, response, event)
            with self.subTest(mutation=mutate), self.assertRaises(corpus.CampaignError):
                sweep.checked_record(case, Path("model.ninfer"), response, event, PRESET)

    def test_decode_windows_exclude_mixed_prefill_and_partial_intervals_not_stalls(self):
        events = [interval(1000, 0, prefill=100), interval(2000, 70, prefill=80),
                  interval(3000, 100), interval(4500, 0, seconds=1.5),
                  interval(5500, 200), interval(6500, 100),
                  interval(7500, 200), interval(8500, 100), interval(9500, 900)]
        events[1]["scheduler"]["prefilling"] = 0  # PP finished during this mixed bin.
        foreign = interval(3500, 99999)
        foreign["server_instance_id"] = "other"
        events.insert(3, foreign)
        result = sweep.decode_windows(events, "owned", 500, 9000)
        self.assertTrue(result["available"])
        self.assertEqual(result["covered_tokens"], 700)
        self.assertEqual(result["covered_seconds"], 6.5)
        self.assertEqual(result["early"]["tokens_per_second"], 40)  # 100/(1+1.5); stall counts.
        self.assertEqual(result["middle"]["tokens_per_second"], 150)
        self.assertEqual(result["late"]["tokens_per_second"], 150)
        self.assertEqual(result["intervals"][0]["estimated_begin_unix_ms"], 2000)

    def test_real_previous_request_tail_does_not_anchor_next_decode_window(self):
        # Literal first failed docs warmup sequence: global counters span both
        # requests. The57 decode tokens belong to the preceding code warmup.
        start, done = 1790116198954, 1790116201331
        rows = [interval(1790116199385, 57, seconds=1.0010897, prefill=1024),
                interval(1790116200387, 105, seconds=1.0022807, prefill=1024)]
        rows[1]["scheduler"]["prefilling"] = 0
        result = sweep.decode_windows(rows, "owned", start, done)
        self.assertFalse(result["available"])
        self.assertEqual(result["covered_tokens"], 0)
        # A later terminal snapshot can precede request_done by1ms; its token
        # delta includes only part of an active second and must not enter TG.
        rows = [interval(1000, 50), interval(2000, 100), interval(3000, 80, running=0)]
        result = sweep.decode_windows(rows, "owned", 0, 3001)
        self.assertEqual(result["covered_tokens"], 100)
        self.assertEqual(len(result["intervals"]), 1)
        concurrent = interval(2000, 100)
        concurrent["scheduler"]["running"] = 2
        with self.assertRaises(corpus.CampaignError):
            sweep.decode_windows([interval(1000, 50), concurrent], "owned", 0, 3001)

    def test_real_logger_delay_is_not_counter_interval_overlap(self):
        # Real docs100K run: JSON timestamps follow steady-clock snapshots and
        # stderr logging. The inferred wall begins overlap by2.0476ms, but the
        # reporter's previous=current contract gives disjoint counter intervals.
        rows = [interval(1790116699747, 60, prefill=100),
                interval(1790116700751, 130, seconds=1.004),
                interval(1790116701753, 128, seconds=1.0040476)]
        rows[0]["scheduler"]["prefilling"] = 0
        result = sweep.decode_windows(rows, "owned", 1790116698000, 1790116702000)
        self.assertEqual(result["covered_tokens"], 258)
        self.assertAlmostEqual(result["covered_seconds"], 2.0080476)
        self.assertEqual(len(result["intervals"]), 2)

    def test_short_generation_has_no_fabricated_windows_and_duplicate_reports_fail(self):
        result = sweep.decode_windows([interval(1000, 100), interval(2000, 100)], "owned", 0, 2200)
        self.assertFalse(result["available"])
        self.assertNotIn("late", result)
        with self.assertRaises(corpus.CampaignError):
            sweep.decode_windows([interval(1000, 100), interval(2000, 100), interval(2000, 100)],
                                 "owned", 0, 3000)
        with self.assertRaises(corpus.CampaignError):
            sweep.decode_windows([interval(1000, 100), interval(2000, 100, prefill=1)], "owned", 0, 3000)

    def test_calibration_preserves_unicode_and_never_repeats_insufficient_source(self):
        body = "\u6d4b\u8bd5\nsource function\n" * 100
        def counter(payload):
            return len(payload["messages"][1]["content"])
        minimum = counter(sweep.make_payload("qwen3.8-27b", "code", "", 4096))
        case = sweep.calibrate("qwen3.8-27b", "code", body,
                               [{"path": "source.cpp", "begin_char": 0, "end_char": len(body)}],
                               minimum + 500, 4096, counter)
        self.assertEqual(case["source_characters"], 500)
        self.assertIn(body[:500], case["payload"]["messages"][1]["content"])
        with self.assertRaises(corpus.CampaignError):
            sweep.calibrate("qwen3.8-27b", "docs", "short", [], 10000, 4096, counter)

    def test_memory_na_is_missing_not_zero_and_summary_is_device_specific(self):
        text = "0, 16303, 14938, 1058, 95, 2500, 14001, 250.1, 66\n1, 16303, 14939, 1057, 96, N/A, 14001, [N/A], 67\n"
        rows = sweep.parse_smi(text)
        self.assertIsNone(rows[1]["clocks.current.sm"])
        self.assertIsNone(rows[1]["power.draw"])
        telemetry = sweep.Telemetry(Path("unused"))
        telemetry.samples = [{"timestamp_unix_ms": 10, "gpus": rows}]
        result = telemetry.summary(0, 20)
        self.assertEqual(result["gpus"][0]["max_used_mib"], 14938)
        self.assertEqual(result["gpus"][1]["min_free_mib"], 1057)
        self.assertIsNone(result["gpus"][1]["max_power_w"])
        with self.assertRaises(ValueError):
            sweep.parse_smi(text.splitlines()[0])

    def test_replay_keeps_exact_bodies_and_rejects_changed_contract(self):
        args = SimpleNamespace(context=196608, prompt_tokens=[180000], completion_tokens=16384,
                               workloads=["code"], repetitions=1)
        cases = []
        for name, phase, tokens, cap in (("warmup-code", "warmup", 2048, 256),
                                         ("code-180000-r1", "requests", 180000, 16384)):
            cases.append({"name": name, "phase": phase, "workload": "code", "target_prompt_tokens": tokens,
                          "expected_prompt_tokens": tokens - 4,
                          "payload": sweep.make_payload("qwen3.8-27b", "code", "Frozen old source\n", cap)})
        frozen = {"artifact_type": "ninfer_context_sweep_workload", "schema_version": 1,
                  "settings": sweep.workload_settings(args), "sampling": PRESET, "cases": cases}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(frozen), encoding="utf-8")
            self.assertEqual(sweep.read_replay(path, args), frozen)
            mutations = [lambda f: f["settings"].update(context=102400),
                         lambda f: f["cases"].reverse(),
                         lambda f: f["cases"][1]["payload"].update(seed=5),
                         lambda f: f["cases"][1].update(expected_prompt_tokens=179000),
                         lambda f: f["cases"][1]["payload"].update(temperature=0)]
            for mutate in mutations:
                changed = copy.deepcopy(frozen)
                mutate(changed)
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(mutation=mutate), self.assertRaises(corpus.CampaignError):
                    sweep.read_replay(path, args)

    def test_request_correlation_rejects_missing_or_external_generation(self):
        done = {**identity("request_done", 2000), "request": {"request_id": 2}}
        matching = {**identity("request_start", 1000), "request": {"request_id": 2}}
        wrong = {**identity("request_start", 900), "request": {"request_id": 1}}
        class Tail:
            def wait_for(self, predicate, description):
                selected = [event for event in (wrong, matching) if predicate(event)]
                if len(selected) != 1:
                    raise AssertionError("ambiguous start correlation")
                return selected[0]
        server = SimpleNamespace(wait_for_request_done=lambda instance: done, tail=Tail())
        self.assertEqual(sweep.correlate(server, "owned", 2), (matching, done))
        with self.assertRaises(corpus.CampaignError):
            sweep.correlate(server, "owned", 1)

    def test_context_and_kv_are_explicit_not_native_rotary_ceiling(self):
        event = {**identity("server_start"),
                 "engine": {**sweep.comparison.EXPECTED_ENGINE, "max_context": 196608, "kv_capacity": 196608,
                            "prefix_reuse": False, "rope_mode": "native", "log_stats_interval_ms": 1000},
                 "server": {"host": sweep.HOST, "port": sweep.PORT, "public_model_id": "qwen3.8-27b"},
                 "artifact": {"target": "qwen3_8_27b", "weights_id": "nvfp4"},
                 "sampling_defaults": {"greedy": False, "non_thinking": PRESET,
                                       "server_overrides": {key: None for key in (*PRESET, "seed")}}}
        self.assertEqual(sweep.validate_start(event, 196608), PRESET)
        for field, value in (("kv_capacity", 102400), ("max_context", 262144),
                             ("effective_max_context", 196608), ("prefix_reuse", True),
                             ("log_stats_interval_ms", 0)):
            changed = copy.deepcopy(event)
            changed["engine"][field] = value
            with self.subTest(field=field), self.assertRaises(corpus.CampaignError):
                sweep.validate_start(changed, 196608)


if __name__ == "__main__":
    unittest.main()
