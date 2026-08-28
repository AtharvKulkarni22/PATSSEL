#!/usr/bin/env python3
"""
LiveCodeBench stdin/stdout verifier-test generation using an RL-trained PEFT Qwen test generator.

This script adapts the original LeetCode assert-style test generator to LiveCodeBench.
Instead of producing:

    def check(candidate):
        assert candidate(...) == ...

it produces JSON stdin/stdout tests:

    {"tests": [{"input": "...", "output": "..."}]}

Because LiveCodeBench does not provide canonical gold solutions, each problem uses:
  1. a full-pass generated prediction from the existing Qwen codegen eval_all files, if available;
  2. otherwise, a buggy farmed solution as a fallback reference.

Generated tests are validated on the selected reference candidate. The final reported
metrics are computed on the farmed buggy solutions, not on gold solutions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


DEFAULT_CODEGEN_ROOT = str(_REPO_ROOT / "data/livecodebench/test.jsonl")
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PEFT_MODEL_PATH = None
DEFAULT_REFERENCE_PRIORITY = ["qwen3_14b", "qwen3_8b", "qwen3_4b", "qwen3_1p7b"]


SYSTEM_PROMPT = """You are a programming-contest verifier-test generation assistant.

Your job is to generate executable stdin/stdout tests for LiveCodeBench-style Python programs.

Output requirements:
- Output ONLY valid JSON.
- Do not output markdown fences.
- Do not output explanations.
- The JSON must be an object with exactly one key: "tests".
- "tests" must be a list of objects.
- Each test object must have exactly these string fields:
  - "input": the complete stdin for one program run.
  - "output": the exact expected stdout for that input.

Test design rules:
- Generate a comprehensive verifier test suite.
- Generate as many tests as needed to cover samples, normal cases, edge cases, boundary cases, tricky cases, and adversarial cases.
- Do not force a fixed number of tests.
- Keep every test deterministic and self-contained.
- Respect the problem's input format exactly.
- Expected outputs must be correct according to the trusted reference candidate.
- Prefer compact tests that each check meaningful behavior.
- Include multi-testcase inputs when the problem format has a leading t.
"""


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def norm_text(s: Any) -> str:
    return textwrap.dedent(str(s or "")).strip()


def normalize_stdout(s: str) -> str:
    # LiveCodeBench generally ignores trailing whitespace/newlines. Keep internal whitespace.
    lines = [ln.rstrip() for ln in (s or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_eval_all_files(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(root.glob("*/*_codegeneration_output_eval_all.json")):
        tag = path.parent.name
        out[tag] = path
    return out


def discover_bug_farms(root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(root.glob("*/*_bug_farm.jsonl")):
        tag = path.parent.name
        out[tag] = path
    return out


def build_reference_and_problem_banks(
    codegen_root: Path,
    reference_priority: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Returns:
      problem_bank[qid] -> problem metadata from eval_all
      full_pass_bank[qid] -> selected full-pass reference candidate
      all_full_pass[qid] -> all full-pass candidates
    """
    eval_files = discover_eval_all_files(codegen_root)
    problem_bank: Dict[str, Dict[str, Any]] = {}
    all_full_pass: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    priority_rank = {tag: i for i, tag in enumerate(reference_priority)}

    for tag, path in eval_files.items():
        data = load_json(path)
        for rec in data:
            qid = str(rec.get("question_id", ""))
            if not qid:
                continue
            problem_bank.setdefault(qid, {
                "question_id": qid,
                "question_title": rec.get("question_title", ""),
                "question_content": rec.get("question_content", ""),
                "platform": rec.get("platform", ""),
                "contest_id": rec.get("contest_id", ""),
                "contest_date": rec.get("contest_date", ""),
                "starter_code": rec.get("starter_code", ""),
                "difficulty": rec.get("difficulty", ""),
            })

            code_list = rec.get("code_list", []) or []
            output_list = rec.get("output_list", code_list) or []
            graded_list = rec.get("graded_list", []) or []
            metadata_list = rec.get("metadata", []) or []

            for idx, code in enumerate(code_list):
                ok = bool(graded_list[idx]) if idx < len(graded_list) else False
                if ok:
                    all_full_pass[qid].append({
                        "question_id": qid,
                        "source_model_tag": tag,
                        "code_idx": idx,
                        "code": code,
                        "raw_output": output_list[idx] if idx < len(output_list) else code,
                        "metadata": metadata_list[idx] if idx < len(metadata_list) else None,
                        "priority_rank": priority_rank.get(tag, 10_000),
                    })

    full_pass_bank: Dict[str, Dict[str, Any]] = {}
    for qid, candidates in all_full_pass.items():
        candidates.sort(key=lambda x: (x["priority_rank"], x["source_model_tag"], x["code_idx"]))
        chosen = dict(candidates[0])
        chosen.pop("priority_rank", None)
        chosen["reference_source"] = "full_pass_prediction"
        full_pass_bank[qid] = chosen

    return problem_bank, full_pass_bank, all_full_pass


def load_buggy_items(codegen_root: Path) -> List[Dict[str, Any]]:
    if codegen_root.is_file():
        return [dict(x) for x in load_jsonl(codegen_root)]
    items: List[Dict[str, Any]] = []
    for tag, path in discover_bug_farms(codegen_root).items():
        for row in load_jsonl(path):
            row = dict(row)
            row.setdefault("base_model_tag", tag)
            items.append(row)
    return items


def select_reference_for_question(
    qid: str,
    full_pass_bank: Dict[str, Dict[str, Any]],
    buggy_by_qid: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if qid in full_pass_bank:
        return full_pass_bank[qid]

    bugs = buggy_by_qid.get(qid, [])
    if not bugs:
        return None

    # Fallback when no full-pass candidate exists. This is lower confidence and recorded.
    b = bugs[0]
    return {
        "question_id": qid,
        "source_model_tag": b.get("base_model_tag", ""),
        "code_idx": b.get("code_idx", 0),
        "code": b.get("buggy_code", ""),
        "raw_output": b.get("buggy_raw_output", b.get("buggy_code", "")),
        "metadata": b.get("metadata", None),
        "reference_source": "fallback_buggy_solution_no_full_pass_available",
    }


def failure_metadata_to_text(metadata: Any, max_chars: int = 4000) -> str:
    if metadata is None:
        return ""
    try:
        s = json.dumps(metadata, ensure_ascii=False, indent=2)
    except Exception:
        s = str(metadata)
    return s[:max_chars]


def build_user_prompt(problem: Dict[str, Any], reference: Dict[str, Any], buggy: Optional[Dict[str, Any]]) -> str:
    buggy_code = norm_text((buggy or {}).get("buggy_code", ""))
    metadata_text = failure_metadata_to_text((buggy or {}).get("metadata", None))
    starter = norm_text(problem.get("starter_code", ""))

    starter_block = starter if starter else "(no starter code; solution should read stdin and write stdout)"

    return f"""Generate verifier tests for this LiveCodeBench programming problem.

Problem title:
{problem.get("question_title", "")}

Question id:
{problem.get("question_id", "")}

Platform:
{problem.get("platform", "")}

Difficulty:
{problem.get("difficulty", "")}

Problem description:
{norm_text(problem.get("question_content", ""))}

Starter code:
```python
{starter_block}
```

Trusted reference candidate:
This is not an official gold solution. It is the best available reference for this run.
Reference source: {reference.get("reference_source", "")}
Reference model/source: {reference.get("source_model_tag", "")}
```python
{norm_text(reference.get("code", ""))}
```

Buggy code sample for context:
The tests should try to expose errors like those in this buggy code, but expected outputs must follow the trusted reference candidate.
```python
{buggy_code}
```

Failure metadata for the buggy code, if available:
```json
{metadata_text}
```

Return ONLY valid JSON in this exact schema:
{{
  "tests": [
    {{
      "input": "complete stdin for one run",
      "output": "exact expected stdout for that input"
    }}
  ]
}}

Generate a comprehensive verifier test suite. Generate as many tests as needed to cover samples, normal cases, edge cases, boundary cases, tricky cases, and adversarial cases. Do not force a fixed number of tests.
"""


def apply_chat_template(tokenizer, prompt: str, device: torch.device):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    return {k: v.to(device) for k, v in inputs.items()}


def load_model_and_tokenizer(args):
    tokenizer_source = args.peft_model_path if args.peft_model_path else args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if torch.cuda.is_available():
        if args.dtype == "bf16":
            dtype = torch.bfloat16
        elif args.dtype == "fp16":
            dtype = torch.float16
        elif args.dtype == "auto":
            dtype = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )

    if args.peft_model_path:
        model = PeftModel.from_pretrained(base_model, args.peft_model_path, is_trainable=False)
        if args.merge_peft:
            model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_raw(model, tokenizer, prompt: str, args) -> str:
    device = next(model.parameters()).device
    inputs = apply_chat_template(tokenizer, prompt, device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if args.temperature <= 0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k

    out = model.generate(**inputs, **gen_kwargs)
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    return raw


def extract_json_obj(raw: str) -> Tuple[Optional[Any], str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty output"

    # Prefer fenced JSON if present.
    matches = JSON_FENCE_RE.findall(text)
    candidates = [m.strip() for m in matches] + [text]

    # Also try substring from first { to last }.
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])

    last_err = ""
    for cand in candidates:
        try:
            return json.loads(cand), ""
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err


def normalize_tests(obj: Any) -> Tuple[List[Dict[str, str]], str]:
    if isinstance(obj, list):
        tests_obj = obj
    elif isinstance(obj, dict) and isinstance(obj.get("tests"), list):
        tests_obj = obj["tests"]
    else:
        return [], "JSON must be an object with key 'tests' containing a list"

    tests: List[Dict[str, str]] = []
    seen = set()
    for t in tests_obj:
        if not isinstance(t, dict):
            continue
        inp = t.get("input", None)
        out = t.get("output", None)
        if not isinstance(inp, str) or not isinstance(out, str):
            continue
        key = (inp, out)
        if key in seen:
            continue
        seen.add(key)
        tests.append({"input": inp, "output": out})

    if not tests:
        return [], "no valid tests after normalization"
    return tests, ""


def run_program_on_input(code: str, stdin_text: str, timeout: float) -> Dict[str, Any]:
    code = norm_text(code)
    if not code:
        return {"ok": False, "stdout": "", "stderr": "empty code", "returncode": None, "timeout": False}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(code + "\n", encoding="utf-8")
        try:
            p = subprocess.run(
                [sys.executable, str(path)],
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=timeout,
                cwd=td,
            )
            return {
                "ok": p.returncode == 0,
                "stdout": p.stdout,
                "stderr": p.stderr,
                "returncode": p.returncode,
                "timeout": False,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "stdout": e.stdout or "",
                "stderr": e.stderr or "timeout",
                "returncode": None,
                "timeout": True,
            }
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": f"{type(e).__name__}: {e}", "returncode": None, "timeout": False}


def evaluate_code_on_tests(code: str, tests: List[Dict[str, str]], timeout: float, keep_details: bool = True) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    passed = 0

    for idx, t in enumerate(tests):
        res = run_program_on_input(code, t["input"], timeout=timeout)
        expected = normalize_stdout(t["output"])
        got = normalize_stdout(res.get("stdout", ""))
        ok = bool(res.get("ok")) and got == expected
        passed += int(ok)

        if keep_details:
            details.append({
                "index": idx,
                "ok": ok,
                "timeout": bool(res.get("timeout", False)),
                "returncode": res.get("returncode"),
                "expected": t["output"],
                "stdout": res.get("stdout", ""),
                "stderr": res.get("stderr", ""),
                "input": t["input"],
            })

    total = len(tests)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / max(1, total),
        "details": details if keep_details else [],
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_completed_qids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("question_id"):
                    done.add(str(rec["question_id"]))
            except Exception:
                pass
    return done


def flush_summary(output_dir: Path, generated_records: List[Dict[str, Any]], buggy_eval_records: List[Dict[str, Any]], args) -> None:
    import csv

    summary_path = output_dir / "summary.tsv"
    fields = [
        "question_id", "question_title", "difficulty", "reference_source", "reference_model",
        "num_raw_generated_tests", "num_valid_generated_tests", "reference_pass_rate_on_raw_tests",
        "num_buggy_solutions", "avg_buggy_pass_rate_on_valid_tests", "bug_detection_rate_on_valid_tests",
    ]

    by_qid_bug = defaultdict(list)
    for r in buggy_eval_records:
        by_qid_bug[str(r.get("question_id", ""))].append(r)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for rec in generated_records:
            qid = str(rec.get("question_id", ""))
            bugs = by_qid_bug.get(qid, [])
            avg_pass = sum(float(b.get("pass_rate_on_valid_tests", 0.0)) for b in bugs) / max(1, len(bugs))
            det_rate = sum(1 for b in bugs if bool(b.get("detected_by_any_valid_test", False))) / max(1, len(bugs))
            w.writerow({
                "question_id": qid,
                "question_title": rec.get("question_title", ""),
                "difficulty": rec.get("difficulty", ""),
                "reference_source": rec.get("reference_source", ""),
                "reference_model": rec.get("reference_model", ""),
                "num_raw_generated_tests": rec.get("num_raw_generated_tests", 0),
                "num_valid_generated_tests": rec.get("num_valid_generated_tests", 0),
                "reference_pass_rate_on_raw_tests": rec.get("reference_pass_rate_on_raw_tests", 0.0),
                "num_buggy_solutions": len(bugs),
                "avg_buggy_pass_rate_on_valid_tests": avg_pass,
                "bug_detection_rate_on_valid_tests": det_rate,
            })

    n = len(generated_records)
    total_bugs = len(buggy_eval_records)
    agg = {
        "model_name": args.model_name,
        "mode": "rl_peft",
        "peft_model_path": str(getattr(args, "peft_model_path", "")),
        "merge_peft": bool(getattr(args, "merge_peft", False)),
        "codegen_root": str(args.codegen_root),
        "num_unique_problems": n,
        "num_buggy_solutions_evaluated": total_bugs,
        "wellformed_rate": sum(bool(r.get("wellformed", False)) for r in generated_records) / max(1, n),
        "avg_num_raw_generated_tests": sum(int(r.get("num_raw_generated_tests", 0)) for r in generated_records) / max(1, n),
        "avg_num_valid_generated_tests": sum(int(r.get("num_valid_generated_tests", 0)) for r in generated_records) / max(1, n),
        "avg_reference_pass_rate_on_raw_tests": sum(float(r.get("reference_pass_rate_on_raw_tests", 0.0)) for r in generated_records) / max(1, n),
        "avg_buggy_pass_rate_on_valid_generated_tests": sum(float(r.get("pass_rate_on_valid_tests", 0.0)) for r in buggy_eval_records) / max(1, total_bugs),
        "avg_bug_detection_rate_on_valid_generated_tests": sum(1 for r in buggy_eval_records if bool(r.get("detected_by_any_valid_test", False))) / max(1, total_bugs),
        "num_fallback_buggy_reference_problems": sum(
            1 for r in generated_records if r.get("reference_source") == "fallback_buggy_solution_no_full_pass_available"
        ),
        "files": {
            "per_problem_generated_tests": str(output_dir / "per_problem_generated_tests.jsonl"),
            "buggy_eval_on_generated_tests": str(output_dir / "buggy_eval_on_generated_tests.jsonl"),
            "summary": str(summary_path),
            "aggregate_metrics": str(output_dir / "aggregate_metrics.json"),
        },
        "config": vars(args),
    }

    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)


def parse_args():
    p = argparse.ArgumentParser(description="Generate LiveCodeBench verifier tests using the base or RL-trained Qwen test generator.")
    p.add_argument("--codegen_root", type=Path, default=Path(DEFAULT_CODEGEN_ROOT), help="Committed LCB bug-farm JSONL (or original bug-farm root).")
    p.add_argument("--reference_root", type=Path, required=True, help="Directory containing the evaluated Qwen *_codegeneration_output_eval_all.json files used to select trusted reference candidates.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--peft_model_path", type=str, default=DEFAULT_PEFT_MODEL_PATH)
    p.add_argument("--merge_peft", action="store_true")
    p.add_argument("--reference_priority", type=str, nargs="+", default=DEFAULT_REFERENCE_PRIORITY)
    p.add_argument("--limit_problems", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--timeout", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=1012)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--trust_remote_code", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    per_problem_path = args.output_dir / "per_problem_generated_tests.jsonl"
    buggy_eval_path = args.output_dir / "buggy_eval_on_generated_tests.jsonl"

    if not args.resume:
        per_problem_path.unlink(missing_ok=True)
        buggy_eval_path.unlink(missing_ok=True)

    print(f"[load] codegen_root={args.codegen_root}", flush=True)
    problem_bank, full_pass_bank, all_full_pass = build_reference_and_problem_banks(args.reference_root, args.reference_priority)
    buggy_items = load_buggy_items(args.codegen_root)

    buggy_by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for b in buggy_items:
        buggy_by_qid[str(b.get("question_id", ""))].append(b)

    qids = sorted(q for q in buggy_by_qid.keys() if q in problem_bank)
    if args.limit_problems is not None:
        qids = qids[: args.limit_problems]

    completed = load_completed_qids(per_problem_path) if args.resume else set()
    qids_to_run = [q for q in qids if q not in completed]

    print(f"[load] unique buggy problem ids={len(qids)} remaining={len(qids_to_run)}", flush=True)
    print(f"[model] loading base={args.model_name} peft={args.peft_model_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args)

    generated_records: List[Dict[str, Any]] = []
    buggy_eval_records: List[Dict[str, Any]] = []

    if args.resume and per_problem_path.exists():
        generated_records = load_jsonl(per_problem_path)
    if args.resume and buggy_eval_path.exists():
        buggy_eval_records = load_jsonl(buggy_eval_path)

    for qid in tqdm(qids_to_run, desc="Generating LCB verifier tests", dynamic_ncols=True):
        problem = problem_bank[qid]
        ref = select_reference_for_question(qid, full_pass_bank, buggy_by_qid)
        if ref is None:
            continue

        representative_bug = buggy_by_qid[qid][0] if buggy_by_qid.get(qid) else None
        prompt = build_user_prompt(problem, ref, representative_bug)

        gen_error = ""
        raw = ""
        parsed_obj = None
        tests: List[Dict[str, str]] = []
        parse_error = ""

        try:
            raw = generate_raw(model, tokenizer, prompt, args)
            parsed_obj, parse_error = extract_json_obj(raw)
            if parsed_obj is not None:
                tests, parse_error = normalize_tests(parsed_obj)
        except Exception as e:
            gen_error = f"{type(e).__name__}: {e}"
            tests = []

        ref_eval_raw = evaluate_code_on_tests(ref.get("code", ""), tests, args.timeout, keep_details=True) if tests else {
            "total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "details": []
        }
        valid_tests = [tests[i] for i, d in enumerate(ref_eval_raw.get("details", [])) if d.get("ok")]

        per_problem_rec = {
            "question_id": qid,
            "question_title": problem.get("question_title", ""),
            "difficulty": problem.get("difficulty", ""),
            "platform": problem.get("platform", ""),
            "reference_source": ref.get("reference_source", ""),
            "reference_model": ref.get("source_model_tag", ""),
            "reference_code_idx": ref.get("code_idx", 0),
            "prompt": prompt,
            "raw_generation": raw,
            "generation_error": gen_error,
            "parse_error": parse_error,
            "wellformed": bool(tests),
            "raw_generated_tests": tests,
            "valid_generated_tests": valid_tests,
            "num_raw_generated_tests": len(tests),
            "num_valid_generated_tests": len(valid_tests),
            "reference_eval_on_raw_tests": ref_eval_raw,
            "reference_pass_rate_on_raw_tests": ref_eval_raw.get("pass_rate", 0.0),
        }
        append_jsonl(per_problem_path, per_problem_rec)
        generated_records.append(per_problem_rec)

        for b in buggy_by_qid.get(qid, []):
            ev = evaluate_code_on_tests(b.get("buggy_code", ""), valid_tests, args.timeout, keep_details=False) if valid_tests else {
                "total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "details": []
            }
            bug_rec = {
                "question_id": qid,
                "base_model_tag": b.get("base_model_tag", ""),
                "code_idx": b.get("code_idx", 0),
                "num_valid_generated_tests": len(valid_tests),
                "passed_valid_tests": ev.get("passed", 0),
                "failed_valid_tests": ev.get("failed", 0),
                "pass_rate_on_valid_tests": ev.get("pass_rate", 0.0),
                "detected_by_any_valid_test": bool(ev.get("failed", 0) > 0),
                "reference_source": ref.get("reference_source", ""),
            }
            append_jsonl(buggy_eval_path, bug_rec)
            buggy_eval_records.append(bug_rec)

        if len(generated_records) % max(1, args.save_every) == 0:
            flush_summary(args.output_dir, generated_records, buggy_eval_records, args)

    flush_summary(args.output_dir, generated_records, buggy_eval_records, args)
    print(f"[done] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

