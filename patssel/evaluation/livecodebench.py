#!/usr/bin/env python3
"""LiveCodeBench data loading and stdin/stdout execution helpers.

The main LiveCodeBench runner handles model inference and experiment routing.
This module keeps benchmark-specific test loading, normalization, execution,
and scoring in one place.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

PASS = "pass"
FAIL = "fail"


def ensure_text(x: Any) -> str:
    """Convert subprocess stdout/stderr to text without changing content."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def normalize_stdout(s: Any) -> str:
    """Normalize line endings and trailing whitespace for stdin/stdout scoring."""
    s = ensure_text(s).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.strip().split("\n")]
    return "\n".join(lines).strip()


def obj_to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def parse_test_cases(x: Any) -> List[Dict[str, str]]:
    """Normalize LCB public/private/generated tests into input/output pairs."""
    if x is None or x == "":
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            return parse_test_cases(json.loads(s))
        except Exception:
            return []
    if isinstance(x, dict):
        if "input" in x and ("output" in x or "expected_output" in x):
            return [{
                "input": str(x.get("input", "")),
                "output": str(x.get("output", x.get("expected_output", ""))),
            }]
        for key in ("tests", "test_cases", "public_test_cases", "private_test_cases"):
            if key in x:
                return parse_test_cases(x[key])
        return []
    if isinstance(x, list):
        out: List[Dict[str, str]] = []
        for item in x:
            if isinstance(item, str):
                out.extend(parse_test_cases(item))
            elif isinstance(item, dict):
                inp = item.get("input", item.get("stdin", item.get("test_input", item.get("input_data", ""))))
                exp = item.get("output", item.get("stdout", item.get("expected_output", item.get("test_output", ""))))
                if inp is not None and exp is not None:
                    out.append({"input": str(inp), "output": str(exp)})
        return out
    return []


def load_lcb_problem_bank(release_version: str) -> Dict[str, Dict[str, Any]]:
    """Load LiveCodeBench problems together with official evaluation tests.

    The loader first checks the normal Hugging Face datasets cache, then tries
    the local LiveCodeBench runner package, and finally Hugging Face datasets.
    No machine-specific cache path is assumed.
    """
    errors = []

    def rows_to_bank(rows: Iterable[Any], source: str) -> Dict[str, Dict[str, Any]]:
        bank: Dict[str, Dict[str, Any]] = {}
        for obj in rows:
            d = obj_to_dict(obj)
            qid = str(d.get("question_id", "") or d.get("task_id", "") or d.get("id", "")).strip()
            if not qid:
                continue

            public_tests = parse_test_cases(d.get("public_test_cases", []))
            private_tests = parse_test_cases(d.get("private_test_cases", []))
            generated_tests = parse_test_cases(d.get("generated_test_cases", []))
            tests = public_tests + private_tests + generated_tests
            if not tests:
                tests = parse_test_cases(d.get("official_tests", []))
            if not tests:
                tests = parse_test_cases(d.get("test_cases", []))
            if not tests:
                tests = parse_test_cases(d.get("tests", []))

            d["official_tests"] = tests
            d["_loaded_from"] = source
            if tests:
                bank[qid] = d
        return bank

    cache_roots = []
    val = os.environ.get("HF_DATASETS_CACHE")
    if val:
        cache_roots.append(Path(val))
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

    seen_roots = []
    for root in cache_roots:
        if root in seen_roots:
            continue
        seen_roots.append(root)
        if not root.exists():
            continue

        patterns = [
            f"livecodebench___code_generation_lite/*version_tag={release_version}*/**/*test*.arrow",
            f"livecodebench___code_generation_lite/*version_tag={release_version}*/**/*.arrow",
            f"livecodebench___code_generation_lite/**/{release_version}/**/*test*.arrow",
            "livecodebench___code_generation_lite/**/*.arrow",
        ]
        arrow_paths = []
        for pat in patterns:
            arrow_paths.extend(sorted(root.glob(pat)))

        dedup = []
        seen = set()
        for ap in arrow_paths:
            if ap not in seen and ap.is_file():
                dedup.append(ap)
                seen.add(ap)

        for ap in dedup:
            try:
                from datasets import Dataset
                ds = Dataset.from_file(str(ap))
                bank = rows_to_bank(list(ds), f"cached_arrow:{ap}")
                if bank:
                    print(f"[lcb] loaded problem bank via cached Arrow: {ap} problems={len(bank)}", flush=True)
                    return bank
                errors.append(f"cached Arrow {ap} had no rows with tests")
            except Exception as e:
                errors.append(f"cached Arrow failed {ap}: {type(e).__name__}: {e}")

    try:
        from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
        rows = list(load_code_generation_dataset(release_version=release_version))
        bank = rows_to_bank(rows, f"lcb_runner release_version={release_version}")
        if bank:
            print(f"[lcb] loaded problem bank via lcb_runner: {len(bank)} problems", flush=True)
            return bank
        errors.append("lcb_runner returned no problems with tests")
    except Exception as e:
        errors.append(f"lcb_runner load failed: {type(e).__name__}: {e}")

    try:
        from datasets import load_dataset
        attempts = [
            ("release_latest", {"version_tag": release_version, "split": "test"}),
            (f"release_latest-version_tag={release_version}", {"split": "test"}),
            ("release_latest", {"split": "test"}),
            (None, {"split": "test"}),
        ]
        for config_name, kwargs in attempts:
            try:
                if config_name is None:
                    ds = load_dataset("livecodebench/code_generation_lite", **kwargs)
                    label = f"datasets default {kwargs}"
                else:
                    ds = load_dataset("livecodebench/code_generation_lite", config_name, **kwargs)
                    label = f"datasets config={config_name} kwargs={kwargs}"
                bank = rows_to_bank(list(ds), label)
                if bank:
                    print(f"[lcb] loaded problem bank via {label}: {len(bank)} problems", flush=True)
                    return bank
                errors.append(f"{label} returned no problems with tests")
            except Exception as e:
                errors.append(f"datasets attempt config={config_name} kwargs={kwargs} failed: {type(e).__name__}: {e}")
    except Exception as e:
        errors.append(f"datasets import/load failed: {type(e).__name__}: {e}")

    raise RuntimeError("Could not load LiveCodeBench problem bank. " + " | ".join(errors))


def run_python_program(code: str, stdin_text: str, timeout: float) -> Dict[str, Any]:
    """Execute one Python stdin/stdout program with a hard timeout."""
    code = code or ""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        path = Path(td) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=stdin_text or "",
                text=True,
                capture_output=True,
                timeout=timeout,
            )
            return {
                "ok_process": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": ensure_text(proc.stdout),
                "stderr": ensure_text(proc.stderr),
                "error_type": "" if proc.returncode == 0 else "NonZeroExit",
            }
        except subprocess.TimeoutExpired as e:
            return {
                "ok_process": False,
                "returncode": None,
                "stdout": ensure_text(e.stdout),
                "stderr": ensure_text(e.stderr),
                "error_type": "Timeout",
                "error_msg": f"timeout >{timeout}s",
            }
        except Exception as e:
            return {
                "ok_process": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "error_type": type(e).__name__,
                "error_msg": str(e),
            }


def evaluate_code_on_tests(
    code: str,
    tests: List[Dict[str, str]],
    timeout: float,
    target_idx: int = 0,
    keep_details: bool = False,
) -> Dict[str, Any]:
    """Evaluate a candidate program on normalized LiveCodeBench tests."""
    details = []
    passed = 0
    total = len(tests)
    for i, t in enumerate(tests):
        inp = str(t.get("input", ""))
        exp = str(t.get("output", ""))
        out = run_python_program(code, inp, timeout)
        got_norm = normalize_stdout(out.get("stdout", ""))
        exp_norm = normalize_stdout(exp)
        ok = bool(out.get("ok_process")) and got_norm == exp_norm
        passed += int(ok)
        if keep_details or i == target_idx:
            details.append({
                "index": i,
                "ok": ok,
                "input": inp if keep_details or i == target_idx else "",
                "expected": exp if keep_details or i == target_idx else "",
                "got": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "error_type": out.get("error_type", ""),
                "error_msg": out.get("error_msg", ""),
            })

    target_ok = False
    target_got = None
    if 0 <= target_idx < total:
        td = next((d for d in details if d.get("index") == target_idx), None)
        if td is None:
            t = tests[target_idx]
            out = run_python_program(code, str(t.get("input", "")), timeout)
            target_ok = bool(out.get("ok_process")) and normalize_stdout(out.get("stdout", "")) == normalize_stdout(str(t.get("output", "")))
            target_got = out.get("stdout", "")
        else:
            target_ok = bool(td.get("ok"))
            target_got = td.get("got")

    return {
        "target_ok": target_ok,
        "target_got": target_got,
        "full_suite_passed": int(passed),
        "full_suite_total": int(total),
        "full_suite_acc_pct": 100.0 * passed / max(1, total),
        "problem_passed": bool(total > 0 and passed == total),
        "status": [PASS if d.get("ok") else FAIL for d in details] if keep_details else [],
        "details": details if keep_details else [],
    }
