"""Run the fixed TP2 quality panel in an owned, temporary Windows server.

Capture reports retain task failures (exit 1). Compare with --baseline to test
exact observable parity separately from task accuracy. Reports are never replaced.
The server is stopped after success, failure, or interruption.
"""

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(r"C:\LLM\neroued\Qwen3.8-27B-nvfp4-NInfer-v2\qwen3_8_27b_nvfp4.ninfer")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--fixtures", type=Path,
                        default=ROOT / "tests/data/quality-panel.json")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/windows/apps/ninfer-serve.exe")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--no-spec", action="store_true", help="Target-only diagnostic control")
    args = parser.parse_args()
    if os.name != "nt" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11 on Windows for this launch configuration")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        raise ValueError("label must contain only letters, numbers, underscores and hyphens")

    output = ROOT / "build/quality"
    output.mkdir(parents=True, exist_ok=True)
    report = output / (args.label + ".json")
    runtime_path = output / (args.label + "-runtime.json")
    launch_path = output / (args.label + "-launch.json")
    log_path = output / (args.label + "-server.log")
    if any(p.exists() for p in (report, runtime_path, launch_path, log_path)):
        raise RuntimeError("Refusing to overwrite an existing stage report")
    with socket.socket() as check:
        check.bind(("127.0.0.1", 19080))

    runtime_options = ["--tp", "2", "--devices", "0,1", "--max-context", "102400",
                       "--kv-capacity", "102400", "--kv-dtype", "int8",
                       "--prefill-chunk", "1024", "--vision", "--image-max-tokens", "2048",
                       "--max-concurrency", "1", "--default-max-tokens", "102400"]
    if not args.no_spec:
        runtime_options += ["--spec", "mtp", "--draft-tokens", "3", "--lm-head-draft"]
    print("Hashing the exact model artifact for the comparison contract...", flush=True)
    runtime = {"artifact_sha256": sha256(args.weights),
               "artifact_bytes": args.weights.stat().st_size, "options": runtime_options}
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    binary = args.binary.resolve(strict=True)
    command = [str(binary), str(args.weights.resolve(strict=True)), *runtime_options,
               "--host", "127.0.0.1", "--port", "19080"]
    launch_path.write_text(json.dumps({"binary": str(binary), "binary_sha256": sha256(binary),
                                      "command": command}, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PATH"] = str(ROOT / "build/windows/vcpkg_installed/x64-windows/bin") + os.pathsep + env["PATH"]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            start = time.monotonic()
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Quality server exited before readiness; see " + str(log_path))
                try:
                    with urlopen("http://127.0.0.1:19080/v1/models", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() - start > 120:
                    raise TimeoutError("Quality server startup timeout")
                time.sleep(0.25)
            print("Quality server ready for " + args.label, flush=True)
            check = [sys.executable, str(ROOT / "tools/test_quality.py"), "--vision",
                     "--label", args.label, "--report", str(report),
                     "--runtime-config", str(runtime_path),
                     "--fixtures", str(args.fixtures.resolve(strict=True))]
            if args.baseline:
                check += ["--baseline", str(args.baseline.resolve(strict=True))]
            result = subprocess.run(check, cwd=ROOT, env=env, timeout=600)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            print("Owned quality server stopped.", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
