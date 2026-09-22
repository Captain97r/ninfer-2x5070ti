"""Exercise false-positive risks in the inference quality gate without a model."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools.test_quality import (comparison_contract, evaluate_case, main, observable,
                                panel_summary, request_body, score, strict_json_loads,
                                validate_prompt_count, validate_reference)


def answer(content, finish="stop", tools=None):
    return {"content": content, "reasoning_content": None, "reasoning": None,
            "tool_calls": tools or [], "finish_reason": finish, "completion_tokens": 4}


class QualityGateTests(unittest.TestCase):
    def test_json_requires_complete_typed_answer(self):
        expected = {"kind": "json", "value": {"count": 1, "found": False}}
        self.assertTrue(score(expected, answer('{"found":false,"count":1}'))[0])
        for text in ('{"count":true,"found":0}', '```json\n{"count":1,"found":false}\n```',
                     '{"count":1,"found":false} extra', '{"count":2,"found":false}',
                     '{"count":9,"count":1,"found":false}', '{"count":NaN,"found":false}'):
            self.assertFalse(score(expected, answer(text))[0], text)
        self.assertFalse(score(expected, answer('{"count":1,"found":false}', "length"))[0])

    def test_tool_requires_exact_name_arguments_and_no_prose(self):
        expected = {"kind": "tool_call", "name": "lookup", "arguments": {"id": 7}}
        call = {"type": "function", "function": {"name": "lookup", "arguments": '{"id":7}'}}
        self.assertTrue(score(expected, answer(None, "tool_calls", [call]))[0])
        for content, calls in (("I will look it up", [call]), (None, [call, call]),
                               (None, [])):
            self.assertFalse(score(expected, answer(content, "tool_calls", calls))[0])
        wrong = copy.deepcopy(call)
        wrong["function"]["arguments"] = '{"id":8}'
        self.assertFalse(score(expected, answer(None, "tool_calls", [wrong]))[0])

    def test_snapshot_ignores_ids_but_preserves_output_and_token_count(self):
        response = {"id": "random", "choices": [{"message": {"content": "ok\n",
                    "reasoning_content": "reason", "tool_calls": [{"id": "random-tool",
                    "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                    "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 4}}
        first = observable(response)
        response["id"] = "another"
        response["choices"][0]["message"]["tool_calls"][0]["id"] = "another-tool"
        self.assertEqual(first, observable(response))
        response["choices"][0]["message"]["content"] = "ok"
        self.assertNotEqual(first, observable(response))
        response["choices"][0]["message"]["content"] = "ok\n"
        response["usage"]["completion_tokens"] = 5
        self.assertNotEqual(first, observable(response))


    @staticmethod
    def run_panel(directory, content, baseline=None, *, expected_prompt_tokens=None, usage=None):
        """Exercise CLI reports and exit codes with HTTP bytes supplied locally."""
        fixtures = directory / "fixtures.json"
        case = {
            "id": "arithmetic", "category": "reasoning",
            "request": {"messages": [{"role": "user", "content": "48-21+10"}]},
            "expected": {"kind": "json", "value": {"remaining": 37}},
        }
        if expected_prompt_tokens is not None:
            case["expected_prompt_tokens"] = expected_prompt_tokens
        fixtures.write_text(json.dumps({"cases": [case]}), encoding="utf-8")
        runtime = directory / "runtime.json"
        runtime.write_text('{"artifact":"fixed.ninfer","spec":"mtp","mtp_k":3}', encoding="utf-8")
        report = directory / ("candidate.json" if baseline else "reference.json")
        args = ["test_quality.py", "--fixtures", str(fixtures), "--runtime-config", str(runtime),
                "--report", str(report)]
        if baseline:
            args.extend(["--baseline", str(baseline)])
        models = {"data": [{"id": "qwen3.8-27b", "context_length": 102400}]}
        response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 4, "prompt_tokens": 11} if usage is None else usage}
        responses = [io.BytesIO(json.dumps(value).encode("utf-8")) for value in (models, response)]
        with patch("sys.argv", args), patch("tools.test_quality.urlopen", side_effect=responses), redirect_stdout(io.StringIO()):
            code = main()
        return code, strict_json_loads(report.read_bytes()), report

    def test_expected_prompt_count_rejects_wrong_context_and_malformed_usage(self):
        for usage in ({"completion_tokens": 4, "prompt_tokens": 12},
                      {"completion_tokens": 4},
                      {"completion_tokens": 4, "prompt_tokens": None},
                      {"completion_tokens": 4, "prompt_tokens": True},
                      {"completion_tokens": 4, "prompt_tokens": 11.0},
                      {"completion_tokens": 4, "prompt_tokens": "11"},
                      {"completion_tokens": 4, "prompt_tokens": -1}):
            with self.subTest(usage=usage), tempfile.TemporaryDirectory() as temp:
                code, report, _ = self.run_panel(
                    Path(temp), '{"remaining":37}', expected_prompt_tokens=11, usage=usage)
                self.assertEqual(code, 1)
                self.assertFalse(report["complete"])
                self.assertEqual(report["cases"][0]["task_status"], "error")
                self.assertEqual(report["cases"][0]["prompt_tokens"], usage.get("prompt_tokens"))
                self.assertIn("error", report["cases"][0])

    def test_prompt_count_metadata_preserves_output_parity_and_checks_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            code, reference, path = self.run_panel(
                directory, '{"remaining":37}', expected_prompt_tokens=11)
            self.assertEqual(code, 0)
            self.assertEqual(reference["cases"][0]["prompt_tokens"], 11)
            self.assertNotIn("prompt_tokens", reference["cases"][0]["observable"])
            code, candidate, _ = self.run_panel(
                directory, '{"remaining":37}', path, expected_prompt_tokens=11)
            self.assertEqual(code, 0)
            self.assertTrue(candidate["all_observables_equal"])
            cases = strict_json_loads((directory / "fixtures.json").read_bytes())["cases"]
            for wrong in (None, 12):
                broken = copy.deepcopy(reference)
                broken["cases"][0]["prompt_tokens"] = wrong
                with self.subTest(reference_count=wrong), self.assertRaises(ValueError):
                    validate_reference(broken, reference["contract"], cases)

    def test_frozen_reference_without_prompt_metadata_still_compares(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, reference, path = self.run_panel(directory, '{"remaining":37}')
            reference["cases"][0].pop("prompt_tokens")
            path.write_text(json.dumps(reference), encoding="utf-8")
            code, candidate, _ = self.run_panel(directory, '{"remaining":37}', path)
            self.assertEqual(code, 0)
            self.assertTrue(candidate["all_observables_equal"])

    def test_expected_prompt_count_requires_an_integer_fixture(self):
        for expected in (None, True, 11.0, "11", -1):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                validate_prompt_count({"expected_prompt_tokens": expected}, 11)

    def test_failed_reference_preserves_failure_but_allows_exact_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            code, reference, path = self.run_panel(directory, '{"remaining":41}')
            self.assertEqual(code, 1)
            self.assertTrue(reference["complete"])
            self.assertFalse(reference["all_tasks_passed"])
            self.assertIsNone(reference["all_observables_equal"])
            self.assertIsNone(reference["regression_passed"])
            self.assertEqual(reference["cases"][0]["task_status"], "failed")
            self.assertNotIn("passed", reference)
            code, candidate, _ = self.run_panel(directory, '{"remaining":41}', path)
            self.assertEqual(code, 0)
            self.assertTrue(candidate["complete"])
            self.assertFalse(candidate["all_tasks_passed"])
            self.assertTrue(candidate["all_observables_equal"])
            self.assertTrue(candidate["no_new_task_failures"])
            self.assertTrue(candidate["regression_passed"])
            row = candidate["cases"][0]
            self.assertEqual(row["task_status"], "failed")
            self.assertEqual(row["regression_status"], "unchanged")
            self.assertFalse(row["rule_passed"])
            self.assertNotIn("passed", row)
            self.assertEqual(candidate["contract"]["runtime_config"],
                             {"artifact": "fixed.ninfer", "spec": "mtp", "mtp_k": 3})

    def test_changed_failed_candidate_cannot_pass_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _, _, path = self.run_panel(directory, '{"remaining":41}')
            code, candidate, _ = self.run_panel(directory, '{"remaining":43}', path)
            self.assertEqual(code, 1)
            self.assertTrue(candidate["complete"])
            self.assertFalse(candidate["all_tasks_passed"])
            self.assertFalse(candidate["all_observables_equal"])
            self.assertTrue(candidate["no_new_task_failures"])
            self.assertFalse(candidate["regression_passed"])
            self.assertEqual(candidate["cases"][0]["regression_status"], "changed")
            # Even an improvement requires review when the contract is exact parity.
            code, improved, _ = self.run_panel(directory, '{"remaining":37}', path)
            self.assertEqual(code, 1)
            self.assertTrue(improved["all_tasks_passed"])
            self.assertFalse(improved["regression_passed"])

    def test_new_task_failure_cannot_pass_regression(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            code, _, path = self.run_panel(directory, '{"remaining":37}')
            self.assertEqual(code, 0)
            code, candidate, _ = self.run_panel(directory, '{"remaining":41}', path)
            self.assertEqual(code, 1)
            self.assertFalse(candidate["no_new_task_failures"])
            self.assertTrue(candidate["cases"][0]["new_task_failure"])

    def test_incomplete_or_errored_reference_is_rejected(self):
        case = {"id": "math", "category": "reasoning",
                "expected": {"kind": "exact_text", "value": "37"}}
        rows = [evaluate_case(case, answer("41"))]
        contract = {"case_ids": ["math"]}
        reference = {"schema_version": 2, "contract": contract, "cases": rows,
                     **panel_summary(rows, ["math"], False)}
        self.assertIn("math", validate_reference(reference, contract, [case]))
        for change in ("missing_case", "error", "incomplete", "missing_answer", "false_score"):
            broken = copy.deepcopy(reference)
            if change == "missing_case":
                broken["cases"] = []
            elif change == "error":
                broken["cases"][0]["error"] = "connection dropped"
            elif change == "incomplete":
                broken["complete"] = False
            elif change == "missing_answer":
                del broken["cases"][0]["observable"]
            else:
                broken["cases"][0]["rule_passed"] = True
                broken["cases"][0]["task_status"] = "passed"
                broken["all_tasks_passed"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_reference(broken, contract, [case])

    def test_image_bytes_and_parsed_runtime_settings_bind_reference(self):
        case = {"id": "vision", "category": "image", "image_fixture": "chart.png",
                "question": "How many?", "request": {"messages": []},
                "expected": {"kind": "exact_text", "value": "2"}}
        settings = {"artifact": "fixed.ninfer", "mtp_k": 3}
        images = {"chart.png": b"original image"}
        contract = comparison_contract(b"fixture", [case], "model", 102400, True, images, settings)
        rows = [evaluate_case(case, answer("2"))]
        reference = {"schema_version": 2, "contract": contract, "cases": rows,
                     **panel_summary(rows, ["vision"], False)}
        reordered = strict_json_loads(' { "mtp_k": 3, "artifact": "fixed.ninfer" } ')
        equivalent = comparison_contract(b"fixture", [case], "model", 102400, True, images, reordered)
        validate_reference(reference, equivalent, [case])
        for image_data, runtime in (
                ({"chart.png": b"changed image"}, settings),
                (images, {"artifact": "fixed.ninfer", "mtp_k": 4})):
            changed = comparison_contract(b"fixture", [case], "model", 102400, True, image_data, runtime)
            with self.assertRaises(ValueError):
                validate_reference(reference, changed, [case])
        body = request_body(case, "model", images)
        self.assertEqual(body["messages"][0]["content"][0]["image_url"]["url"],
                         "data:image/png;base64,b3JpZ2luYWwgaW1hZ2U=")

    def test_strict_json_rejects_nonfinite_numbers_including_overflow(self):
        for number in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            with self.subTest(number=number), self.assertRaises(ValueError):
                strict_json_loads('{"nested":[' + number + "]}")
        self.assertEqual(strict_json_loads('{"value":1e308}'), {"value": 1e308})


if __name__ == "__main__":
    unittest.main()
