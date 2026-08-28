#!/usr/bin/env python3
"""
Unified HE+Fix table runner.

This is the HE+Fix analogue of leetcode_table_unified_runner.py. It runs the
same experiment families over archiki/UTGenDebug / he_plus_fix, but uses the
HE+Fix data contract:

  row["prompt"]              : function header + docstring / problem prompt
  row["code"]                : buggy function body or full buggy function
  row["canonical_solution"]  : gold/reference full function
  row["entry_point"]         : callable function name
  row["test"]                : oracle/reference tests containing inputs/results

Oracle/reference tests are extracted from row["test"] and executed with
UTGenDebug's unsafe_execute. Generated verifier tests are loaded from the
HE+Fix test-generator JSONL files and used ONLY for routing; final reported
pass rates are always computed on the oracle/reference tests from the dataset.

The minimal HE+Fix execution helpers are provided in patssel/evaluation/hefix.py.

Example
-------
  python scripts/run_hefix.py --all --log_dir outputs/results/hefix
"""

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import ast
import csv
import json
import random
import shutil
import argparse
import textwrap
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# Minimal HE+Fix execution utilities extracted from the original experiment code.
from patssel.evaluation.hefix import unsafe_execute, IMPORT_HELPER

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

DEFAULT_SEED = 1012
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_DATASET_NAME = "archiki/UTGenDebug"
DEFAULT_SPLIT = "he_plus_fix"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_TIMEOUT = 60
DEFAULT_BASE_TESTGEN_JSONL = str(_REPO_ROOT / "outputs/verifier_tests/hefix/base/per_problem_detailed.jsonl")
DEFAULT_RL_TESTGEN_JSONL = str(_REPO_ROOT / "outputs/verifier_tests/hefix/rl/per_problem_detailed.jsonl")
DEFAULT_RL_PEFT_CHECKPOINT = str(_REPO_ROOT / "checkpoints/debugger")

FENCE_PY_RE = re.compile(r"```python\s*([\s\S]*?)\s*```", re.IGNORECASE)
GENERIC_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+\-]*\s*\n([\s\S]*?)\n```$", re.S)


def expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p or ""))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def extract_fenced_code(text: str) -> str:
    """Return LAST ```python ...``` block only. Empty string if none."""
    if not text:
        return ""
    matches = FENCE_PY_RE.findall(text)
    return matches[-1].strip() if matches else ""


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    m = GENERIC_FENCE_RE.match(t)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+\-]*\s*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def extract_generation_code(text: str) -> str:
    fenced = extract_fenced_code(text)
    return fenced if fenced else strip_code_fences(text)


def looks_like_toplevel_def_or_class(code: str) -> bool:
    t = (code or "").lstrip()
    return t.startswith("def ") or t.startswith("class ")


def indent_as_block(body: str, spaces: int = 4) -> str:
    body = normalize_newlines(body).strip("\n")
    if not body.strip():
        return ""
    lines = body.split("\n")
    already_indented = any(
        ln.strip() != "" and (ln.startswith(" ") or ln.startswith("\t"))
        for ln in lines
    )
    if already_indented:
        return body
    pad = " " * spaces
    return "\n".join((pad + ln) if ln.strip() else ln for ln in lines)


def build_program_source_from_prompt_code(prompt: str, code: str) -> str:
    """
    HE+Fix code is often a function body. Build executable buggy source as
    prompt + indented(body), unless code already contains a top-level def/class.
    """
    prompt = normalize_newlines(prompt).rstrip() + "\n"
    code = normalize_newlines(code).rstrip()
    if looks_like_toplevel_def_or_class(code):
        return code + "\n"
    return prompt + indent_as_block(code, spaces=4) + "\n"


def first_shard_device(model) -> str:
    dmap = getattr(model, "hf_device_map", None)
    if not dmap:
        return str(next(model.parameters()).device)
    for dev in dmap.values():
        if isinstance(dev, int):
            return f"cuda:{dev}"
        if isinstance(dev, str) and dev.startswith("cuda"):
            return dev
    return "cpu"


# -------------------------
# HE+Fix oracle test extraction and execution
# -------------------------
def extract_inputs_results_from_test(test_code: str) -> Tuple[Optional[List[Any]], Optional[List[Any]], bool]:
    """
    HE+Fix test field defines inputs/results inside def check(candidate).
    Extract those two lists with AST and execute only those assignments.
    """
    test_code = normalize_newlines(test_code)
    use_set = (" set(" in test_code)
    try:
        tree = ast.parse(test_code)
    except Exception:
        return None, None, use_set

    check_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_fn = node
            break
    if check_fn is None:
        return None, None, use_set

    assigns_src: List[str] = []
    for stmt in check_fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            if stmt.targets[0].id in ("inputs", "results"):
                seg = ast.get_source_segment(test_code, stmt)
                if seg:
                    assigns_src.append(seg)
    if not assigns_src:
        return None, None, use_set

    sandbox: Dict[str, Any] = {}
    try:
        exec("import numpy as np\n", sandbox, sandbox)
        exec("\n".join(assigns_src), sandbox, sandbox)
        inputs = sandbox.get("inputs")
        results = sandbox.get("results")
        if not isinstance(inputs, list) or not isinstance(results, list) or len(inputs) != len(results):
            return None, None, use_set
        return inputs, results, use_set
    except Exception:
        return None, None, use_set


def unsafe_execute_wrapper(entry_point: str, code: str, unit_inputs: List[Any], unit_outputs: List[Any], verbose: bool, use_set: bool):
    try:
        return unsafe_execute("mbpp", entry_point, code, unit_inputs, unit_outputs, verbose=verbose, use_set=use_set)
    except Exception:
        return None


def run_unsafe_execute(entry_point: str, code: str, unit_inputs: List[Any], unit_outputs: List[Any], use_set: bool, timeout: int, verbose: bool = True) -> Tuple[List[str], List[Any]]:
    n = len(unit_inputs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(unsafe_execute_wrapper, entry_point, code, unit_inputs, unit_outputs, verbose, use_set)
        try:
            result = fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            return [TIMEOUT] * n, [f"unsafe_execute timeout >{timeout}s"] * n
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            return [TIMEOUT] * n, ["unsafe_execute crashed"] * n
    if result is None:
        return [TIMEOUT] * n, ["unsafe_execute returned None"] * n
    try:
        pass_list, exec_out = result
        if not isinstance(pass_list, list) or len(pass_list) != n:
            return [TIMEOUT] * n, ["bad unsafe_execute output"] * n
        if not isinstance(exec_out, list) or len(exec_out) != n:
            exec_out = [""] * n
        return pass_list, exec_out
    except Exception:
        return [TIMEOUT] * n, ["could not unpack unsafe_execute result"] * n


def evaluate_oracle_candidate(entry_point: str, candidate_code: str, unit_inputs: List[Any], unit_outputs: List[Any], use_set: bool, timeout: int, target_idx: int) -> Dict[str, Any]:
    task_setup = "\n".join(IMPORT_HELPER["python"])
    status, exec_out = run_unsafe_execute(
        entry_point=entry_point,
        code=task_setup + "\n" + (candidate_code or ""),
        unit_inputs=unit_inputs,
        unit_outputs=unit_outputs,
        use_set=use_set,
        timeout=timeout,
        verbose=True,
    )
    total = len(status)
    passed = int(sum(1 for s in status if s == PASS))
    return {
        "target_ok": bool(0 <= target_idx < total and status[target_idx] == PASS),
        "target_got": exec_out[target_idx] if 0 <= target_idx < len(exec_out) else None,
        "full_suite_passed": passed,
        "full_suite_total": total,
        "full_suite_acc_pct": (100.0 * passed / max(1, total)) if total else 0.0,
        "status": status,
        "exec_out": exec_out,
    }


# -------------------------
# Generated verifier-test execution
# -------------------------
class _CheckToCollector(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name != "check":
            return node
        node.name = "__collect_asserts"
        new_body = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assert):
                lam = ast.Lambda(
                    args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                    body=stmt.test,
                )
                call = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="__ASSERTS__", ctx=ast.Load()), attr="append", ctx=ast.Load()),
                    args=[lam],
                    keywords=[],
                )
                new_body.append(ast.copy_location(ast.Expr(value=call), stmt))
        node.body = new_body if new_body else [ast.Pass()]
        return node


def count_asserts(test_code: str) -> int:
    try:
        tree = ast.parse(test_code or "")
        return sum(isinstance(n, ast.Assert) for n in ast.walk(tree))
    except Exception:
        return 0


def thunkify_tests(test_code: str) -> str:
    tree = ast.parse(test_code or "")
    tree = _CheckToCollector().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _verifier_worker(q, entry_point: str, candidate_code: str, verifier_test_code: str):
    try:
        g: Dict[str, Any] = {"__name__": "__main__"}
        # Broad but standard helper prelude used by generated HE+Fix tests.
        prelude = """
import math
import random
import itertools
import functools
import collections
import heapq
import bisect
import re
import string
from typing import *
try:
    import numpy as np
except Exception:
    np = None
"""
        exec(prelude, g, g)
        exec("\n".join(IMPORT_HELPER["python"]), g, g)
        exec(candidate_code or "", g, g)
        candidate = g.get(entry_point)
        if candidate is None:
            raise RuntimeError(f"entry point not found: {entry_point}")
        g["__ASSERTS__"] = []
        transformed = thunkify_tests(verifier_test_code)
        exec(transformed, g, g)
        g["__collect_asserts"](candidate)
        thunks = list(g.get("__ASSERTS__", []))
        results = []
        for i, th in enumerate(thunks):
            try:
                ok = bool(th())
                # For assert candidate(...) == ... rewritten to lambda: comparison,
                # False means assertion would fail.
                results.append({"index": i, "ok": ok, "error_type": "" if ok else "AssertionError"})
            except AssertionError as e:
                results.append({"index": i, "ok": False, "error_type": "AssertionError", "error_msg": str(e)})
            except Exception as e:
                results.append({"index": i, "ok": False, "error_type": type(e).__name__, "error_msg": str(e)})
        q.put({"ok": True, "results": results, "total": len(thunks), "error": ""})
    except Exception as e:
        total = count_asserts(verifier_test_code)
        q.put({
            "ok": False,
            "results": [{"index": i, "ok": False, "error_type": type(e).__name__, "error_msg": str(e)} for i in range(total)],
            "total": total,
            "error": f"{type(e).__name__}: {e}",
        })


def evaluate_verifier_candidate(entry_point: str, candidate_code: str, verifier_test_code: str, timeout: int, target_idx: int = 0) -> Dict[str, Any]:
    q = None
    import multiprocessing as mp
    q = mp.Queue()
    p = mp.Process(target=_verifier_worker, args=(q, entry_point, candidate_code, verifier_test_code))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(1)
        total = count_asserts(verifier_test_code)
        return {
            "target_ok": False,
            "full_suite_passed": 0,
            "full_suite_total": total,
            "full_suite_acc_pct": 0.0,
            "results": [],
            "error": f"timeout >{timeout}s",
        }
    if q.empty():
        total = count_asserts(verifier_test_code)
        return {"target_ok": False, "full_suite_passed": 0, "full_suite_total": total, "full_suite_acc_pct": 0.0, "results": [], "error": "no result"}
    out = q.get()
    results = out.get("results", [])
    total = int(out.get("total", len(results)))
    passed = int(sum(1 for r in results if r.get("ok")))
    return {
        "target_ok": bool(0 <= target_idx < len(results) and results[target_idx].get("ok")),
        "full_suite_passed": passed,
        "full_suite_total": total,
        "full_suite_acc_pct": (100.0 * passed / max(1, total)) if total else 0.0,
        "results": results,
        "error": out.get("error", ""),
    }


# -------------------------
# Prompt builders
# -------------------------
def build_debug_messages(buggy_or_current_code: str, failing_input: Any, failing_expected: Any, failing_got: Any, entry_point_hint: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are a Python bug-fixing assistant.\n\n"
        "You MUST respond with the exact structure below. Every section is REQUIRED.\n\n"
        "1) CODE INTENT\n"
        "2) BUG REVIEW\n"
        "3) FIX PLAN\n"
        "4) FINAL FIX\n"
        "   Output ONLY the fixed function/class in exactly ONE Python fenced code block.\n"
        "   The code block MUST be the LAST thing in the response.\n"
        "   Keep the same function/class name and signature. Do NOT change imports.\n"
    )
    user_msg = (
        "Fix the buggy function/class so that the failing unit test passes, and try to fix any other issues in the code.\n"
        "- Keep the same function/class name and signature.\n"
        "- Do NOT change imports.\n"
        "- The final answer must end with exactly one Python fenced code block.\n\n"
        "Buggy function/class:\n"
        f"{textwrap.dedent(buggy_or_current_code).strip()}\n\n"
        "Failing unit test:\n"
        f"Input: {repr(failing_input)}\n"
        f"Expected: {repr(failing_expected)}\n"
        f"Got: {repr(failing_got)}\n\n"
        f"Entry point: {entry_point_hint or '(unknown)'}\n"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def build_generation_messages(prompt: str, signature: str, entry_point: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are an expert Python programmer. Solve the programming task.\n"
        "Return exactly one Python fenced code block containing the full function/class implementation.\n"
        "Keep the requested function name and signature. Do not include explanations outside the code block."
    )
    user_msg = (
        "Generate a correct Python solution for this task.\n\n"
        "Problem prompt / starter code:\n"
        f"{prompt}\n\n"
        f"Signature: {signature or '(see prompt)'}\n"
        f"Entry point: {entry_point}\n"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


# -------------------------
# Model wrapper
# -------------------------
class ModelRunner:
    def __init__(self, model, tokenizer, model_id: str, peft_checkpoint: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.peft_checkpoint = peft_checkpoint
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def render_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    @torch.no_grad()
    def generate_batch(self, batch_messages: List[List[Dict[str, str]]], deterministic: bool, max_new_tokens: int, temperature: float, top_p: float) -> List[Tuple[str, int]]:
        if not batch_messages:
            return []
        prompt_texts = [self.render_chat_prompt(m) for m in batch_messages]
        prompt_lens = [len(json.dumps(m, ensure_ascii=False)) for m in batch_messages]
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=False)
        dev = first_shard_device(self.model)
        enc = {k: v.to(dev) for k, v in enc.items()}
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
        if deterministic:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        outputs = self.model.generate(**enc, **gen_kwargs)
        results = []
        for i in range(outputs.shape[0]):
            gen_ids = outputs[i][input_lengths[i]:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append((gen_text, prompt_lens[i]))
        return results


def _dtype_from_string(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def load_model_and_tokenizer(model_id: str, peft_checkpoint: Optional[str], dtype: str) -> ModelRunner:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    torch_dtype = _dtype_from_string(dtype)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    if peft_checkpoint:
        if PeftModel is None:
            raise RuntimeError("peft is not available, but --rl_peft_checkpoint was provided")
        model = PeftModel.from_pretrained(base, peft_checkpoint)
    else:
        model = base
    model.eval()
    return ModelRunner(model=model, tokenizer=tokenizer, model_id=model_id, peft_checkpoint=peft_checkpoint)


def free_runner(runner: Optional[ModelRunner]) -> None:
    if runner is None:
        return
    try:
        del runner.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -------------------------
# Dataset and verifier loading
# -------------------------
def load_hefix_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    ds = load_dataset(args.dataset_name, split=args.split)
    rows = [dict(x) for x in ds]

    if args.limit_tasks and args.limit_tasks > 0:
        rows = rows[:args.limit_tasks]

    num_shards = int(getattr(args, "num_shards", 1) or 1)
    shard_idx = int(getattr(args, "shard_idx", 0) or 0)

    if num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"--shard_idx must be in [0, {num_shards}), got {shard_idx}")

    if num_shards > 1:
        rows = [row for i, row in enumerate(rows) if i % num_shards == shard_idx]

    return rows


def load_verifier_tests_map(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    jsonl_path = expand_path(jsonl_path)
    if not jsonl_path:
        return out
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            task_id = str(rec.get("task_id", "")).strip()
            if not task_id:
                continue
            code = str(
                rec.get("generated_test_code_eval", "")
                or rec.get("generated_test_code_clean", "")
                or rec.get("generated_test_code_raw_extracted", "")
                or ""
            )
            out[task_id] = {
                "verifier_test_code": code,
                "num_asserts": int(rec.get("num_eval_asserts", 0) or rec.get("num_generated_asserts", 0) or count_asserts(code)),
                "record": rec,
            }
    return out


# -------------------------
# Experiment definitions
# -------------------------
@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str  # buggy, single_debug, single_generation, best2_generation_oracle, best2_debug_oracle, routing_oracle, routing_verifier
    debug_model: Optional[str] = None  # base, rl
    generation_model: Optional[str] = None  # base, rl
    verifier_source: Optional[str] = None  # base_testgen, rl_testgen
    table_label: str = ""


EXPERIMENTS: Dict[str, Experiment] = {
    "base_regen": Experiment("base_regen", "best2_generation_verifier", generation_model="base", verifier_source="base_testgen", table_label="Base ReGen"),
    "base_debug": Experiment("base_debug", "best2_debug_verifier", debug_model="rl", verifier_source="base_testgen", table_label="Base Debug"),
    "base_patssel": Experiment("base_patssel", "routing_verifier", debug_model="rl", generation_model="base", verifier_source="base_testgen", table_label="Base PATSSEL"),
    "rl_regen": Experiment("rl_regen", "best2_generation_verifier", generation_model="base", verifier_source="rl_testgen", table_label="RL ReGen"),
    "rl_debug": Experiment("rl_debug", "best2_debug_verifier", debug_model="rl", verifier_source="rl_testgen", table_label="RL Debug"),
    "rl_patssel": Experiment("rl_patssel", "routing_verifier", debug_model="rl", generation_model="base", verifier_source="rl_testgen", table_label="RL PATSSEL"),
    "gold_regen": Experiment("gold_regen", "best2_generation_oracle", generation_model="base", table_label="Gold ReGen"),
    "gold_debug": Experiment("gold_debug", "best2_debug_oracle", debug_model="rl", table_label="Gold Debug"),
    "gold_patssel": Experiment("gold_patssel", "routing_oracle", debug_model="rl", generation_model="base", table_label="Gold PATSSEL"),
}

ALL_EXPERIMENT_ORDER = [
    "base_regen", "base_debug", "base_patssel",
    "rl_regen", "rl_debug", "rl_patssel",
    "gold_regen", "gold_debug", "gold_patssel",
]


def choose_candidate(debug_eval: Dict[str, Any], gen_eval: Dict[str, Any]) -> str:
    if gen_eval["full_suite_passed"] > debug_eval["full_suite_passed"]:
        return "generation"
    if gen_eval["full_suite_passed"] < debug_eval["full_suite_passed"]:
        return "debug"
    if gen_eval.get("target_ok") and not debug_eval.get("target_ok"):
        return "generation"
    return "debug"


def choose_best_oracle_candidate(candidate_evals: List[Dict[str, Any]]) -> int:
    """Return the candidate index with the highest oracle/reference-test score.

    Ties are broken by whether the selected failing target is fixed, then by
    earliest sample. This mirrors the LeetCode best-of-2 runner.
    """
    if not candidate_evals:
        raise ValueError("candidate_evals must be non-empty")
    best_idx = 0
    best_key = (
        int(candidate_evals[0].get("full_suite_passed", 0)),
        float(candidate_evals[0].get("full_suite_acc_pct", 0.0)),
        int(bool(candidate_evals[0].get("target_ok", False))),
    )
    for i, ev in enumerate(candidate_evals[1:], start=1):
        key = (
            int(ev.get("full_suite_passed", 0)),
            float(ev.get("full_suite_acc_pct", 0.0)),
            int(bool(ev.get("target_ok", False))),
        )
        if key > best_key:
            best_idx = i
            best_key = key
    return best_idx


def choose_best_verifier_candidate(candidate_evals: List[Dict[str, Any]]) -> int:
    """Return candidate index with highest generated-verifier-test score."""
    if not candidate_evals:
        raise ValueError("candidate_evals must be non-empty")
    best_idx = 0
    best_key = (
        int(candidate_evals[0].get("full_suite_passed", 0)),
        float(candidate_evals[0].get("full_suite_acc_pct", 0.0)),
        int(bool(candidate_evals[0].get("target_ok", False))),
    )
    for i, ev in enumerate(candidate_evals[1:], start=1):
        key = (
            int(ev.get("full_suite_passed", 0)),
            float(ev.get("full_suite_acc_pct", 0.0)),
            int(bool(ev.get("target_ok", False))),
        )
        if key > best_key:
            best_idx = i
            best_key = key
    return best_idx


def summarize_candidate_evals(prefix: str, evals: List[Dict[str, Any]], selected_idx: int) -> str:
    parts = []
    for i, ev in enumerate(evals, start=1):
        parts.append(
            f"{prefix}{i}={int(ev.get('full_suite_passed', 0))}/"
            f"{int(ev.get('full_suite_total', 0))}"
            f"({float(ev.get('full_suite_acc_pct', 0.0)):.2f}%)"
        )
    parts.append(f"selected_{prefix}{selected_idx + 1}")
    return ";".join(parts)


def public_eval(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not d:
        return None
    return {k: v for k, v in d.items() if k not in ("status", "exec_out", "results")}


# -------------------------
# Core experiment runner
# -------------------------
def prepare_row(ex: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    task_id = str(ex.get("task_id", "")).strip()
    entry_point = str(ex.get("entry_point", "")).strip()
    prompt = str(ex.get("prompt", "") or "")
    buggy_body_or_code = str(ex.get("code", "") or "")
    canonical_solution = str(ex.get("canonical_solution", "") or "")
    test_code = str(ex.get("test", "") or "")
    signature = str(ex.get("signature", "") or "")

    unit_inputs, unit_outputs, use_set = extract_inputs_results_from_test(test_code)
    if not task_id or not entry_point or unit_inputs is None or unit_outputs is None or len(unit_inputs) == 0:
        return None

    buggy_code = build_program_source_from_prompt_code(prompt, buggy_body_or_code).strip()
    baseline = evaluate_oracle_candidate(entry_point, buggy_code, unit_inputs, unit_outputs, use_set, args.timeout, target_idx=0)
    failing_idxs = [i for i, s in enumerate(baseline["status"]) if s != PASS]
    target_idx = int(failing_idxs[0]) if args.target_policy == "first" and failing_idxs else int(random.choice(failing_idxs)) if failing_idxs else -1
    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "prompt": prompt,
        "signature": signature,
        "buggy_code": buggy_code,
        "canonical_solution": canonical_solution,
        "unit_inputs": unit_inputs,
        "unit_outputs": unit_outputs,
        "use_set": use_set,
        "baseline_eval": baseline,
        "target_idx": target_idx,
        "target_input": unit_inputs[target_idx] if target_idx >= 0 else None,
        "target_expected": unit_outputs[target_idx] if target_idx >= 0 else None,
        "target_got_baseline": baseline["exec_out"][target_idx] if target_idx >= 0 and target_idx < len(baseline["exec_out"]) else None,
    }


def _metric_int(d: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(d.get(key, default))
    except Exception:
        return default


def _metric_bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    try:
        return bool(d.get(key, default))
    except Exception:
        return default


def _add_count(d: Dict[str, int], key: str) -> None:
    if not key:
        key = "unknown"
    d[key] = d.get(key, 0) + 1


def _log_has_content(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _open_tsv_for_resume(path: str, header: List[str], force: bool) -> Tuple[Any, csv.writer]:
    """Open a TSV in append mode and write its header only when creating it."""
    if force and os.path.exists(path):
        os.remove(path)
    need_header = not _log_has_content(path)
    fp = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fp, delimiter="\t")
    if need_header:
        writer.writerow(header)
        fp.flush()
    return fp, writer


def load_resume_state(summary_jsonl: str, experiment_name: str) -> Dict[str, Any]:
    """
    Load row-level resume state from summary_per_problem.jsonl.

    We treat a task_id as completed only after a complete JSONL record has been
    written. If the job dies while a row is still in memory or mid-generation,
    that row is rerun, which is the safest behavior.
    """
    by_task: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(summary_jsonl):
        return {
            "completed_ids": set(),
            "evaluated": 0,
            "skipped": 0,
            "nb_passed": 0,
            "nb_total": 0,
            "nb_target": 0,
            "bt_passed": 0,
            "bt_total": 0,
            "bt_target": 0,
            "nb_counts": {},
            "bt_counts": {},
        }

    with open(summary_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # Ignore a torn/partial line from an interrupted write.
                continue
            if rec.get("experiment") not in (None, "", experiment_name):
                continue
            tid = str(rec.get("task_id", "")).strip()
            if not tid:
                continue
            event = rec.get("event", "")
            if event in ("final_metrics", "skipped_prepare_row"):
                # Last complete record wins, avoiding double counting if a user
                # manually appended duplicate task records.
                by_task[tid] = rec

    state = {
        "completed_ids": set(by_task.keys()),
        "evaluated": 0,
        "skipped": 0,
        "nb_passed": 0,
        "nb_total": 0,
        "nb_target": 0,
        "bt_passed": 0,
        "bt_total": 0,
        "bt_target": 0,
        "nb_counts": {},
        "bt_counts": {},
    }

    for rec in by_task.values():
        if rec.get("event") == "skipped_prepare_row":
            state["skipped"] += 1
            continue
        if rec.get("event") != "final_metrics":
            continue
        nb_eval = rec.get("nonbacktracking_eval", {}) or {}
        bt_eval = rec.get("backtracking_eval", {}) or {}
        state["evaluated"] += 1
        state["nb_passed"] += _metric_int(nb_eval, "full_suite_passed")
        state["nb_total"] += _metric_int(nb_eval, "full_suite_total")
        state["nb_target"] += int(_metric_bool(nb_eval, "target_ok"))
        state["bt_passed"] += _metric_int(bt_eval, "full_suite_passed")
        state["bt_total"] += _metric_int(bt_eval, "full_suite_total")
        state["bt_target"] += int(_metric_bool(bt_eval, "target_ok"))
        _add_count(state["nb_counts"], str(rec.get("selected_source", "unknown")))
        _add_count(state["bt_counts"], str(rec.get("backtracking_source", "unknown")))

    return state


def build_final_metrics(experiment: Experiment, mode: str, problems_seen: int, evaluated: int, skipped: int, passed: int, total: int, target_fixed: int, source_counts: Dict[str, int], args: argparse.Namespace) -> Dict[str, Any]:
    source_counts_clean = {k: int(v) for k, v in sorted(source_counts.items())}
    return {
        "experiment": experiment.name,
        "table_label": experiment.table_label,
        "kind": experiment.kind,
        "dataset": args.dataset_name,
        "split": args.split,
        "mode": mode,
        "problems_seen": int(problems_seen),
        "evaluated": int(evaluated),
        "skipped_count": int(skipped),
        "final_full_suite_passed_total": int(passed),
        "final_full_suite_total_asserts": int(total),
        "overall_accuracy_pct": float(100.0 * passed / max(1, total)) if total else 0.0,
        "targeted_fixed_count": int(target_fixed),
        "targeted_fixed_pct": float(100.0 * target_fixed / max(1, evaluated)) if evaluated else 0.0,
        "generation_count": int(sum(v for k, v in source_counts_clean.items() if str(k).startswith("generation"))),
        "debug_count": int(sum(v for k, v in source_counts_clean.items() if str(k).startswith("debug"))),
        "buggy_count": int(source_counts_clean.get("buggy", 0)),
        "source_counts": source_counts_clean,
        "seed": int(args.seed),
        "target_policy": args.target_policy,
        "deterministic_generation": bool(args.deterministic_test),
        "gen_params": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k},
        "timeout": int(args.timeout),
    }


def run_experiment(experiment: Experiment, args: argparse.Namespace, rows: List[Dict[str, Any]], runners: Dict[str, ModelRunner], verifier_maps: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    root = os.path.join(args.log_dir, experiment.name)
    out_non = os.path.join(root, "non_backtracking")
    out_bt = os.path.join(root, "backtracking")
    ensure_dir(out_non); ensure_dir(out_bt)

    final_non_path = os.path.join(out_non, "final_metrics.json")
    final_bt_path = os.path.join(out_bt, "final_metrics.json")
    final_all_path = os.path.join(root, "final_metrics_all.json")

    if (not args.force) and os.path.exists(final_non_path) and os.path.exists(final_bt_path):
        print(f"[cache] {experiment.name}: loading final metrics")
        if os.path.exists(final_all_path):
            with open(final_all_path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        with open(final_non_path, "r", encoding="utf-8") as fp:
            final_non = json.load(fp)
        with open(final_bt_path, "r", encoding="utf-8") as fp:
            final_bt = json.load(fp)
        final_all = {"experiment": experiment.name, "table_label": experiment.table_label, "non_backtracking": final_non, "backtracking": final_bt}
        with open(final_all_path, "w", encoding="utf-8") as fp:
            json.dump(final_all, fp, ensure_ascii=False, indent=2)
        return final_all

    attempts_tsv = os.path.join(out_non, "attempts_log_with_full_code_and_suite.tsv")
    baseline_tsv = os.path.join(out_non, "baseline_metrics.tsv")
    summary_jsonl = os.path.join(out_non, "summary_per_problem.jsonl")

    if args.force:
        for p in (attempts_tsv, baseline_tsv, summary_jsonl, final_non_path, final_bt_path, final_all_path):
            if os.path.exists(p):
                os.remove(p)

    resume_state = load_resume_state(summary_jsonl, experiment.name) if not args.force else load_resume_state("/path/that/does/not/exist", experiment.name)
    completed_ids = set(resume_state["completed_ids"])
    if completed_ids:
        print(f"[resume] {experiment.name}: found {len(completed_ids)} completed task rows; continuing from partial logs")

    attempts_header = [
        "experiment", "task_id", "target_idx",
        "baseline_passed", "baseline_total", "baseline_acc_pct",
        "debug_passed", "debug_total", "debug_acc_pct",
        "generation_passed", "generation_total", "generation_acc_pct",
        "verifier_debug_passed", "verifier_debug_total", "verifier_generation_passed", "verifier_generation_total",
        "selected_source", "nonbacktracking_passed", "nonbacktracking_total", "nonbacktracking_acc_pct",
        "backtracking_source", "backtracking_passed", "backtracking_total", "backtracking_acc_pct",
        "target_ok", "debug_prompt_chars", "generation_prompt_chars", "debug_output_chars", "generation_output_chars",
        "target_input", "target_expected", "target_got_baseline", "target_got_selected",
        "note", "debug_code_full", "generation_code_full", "selected_code_full",
    ]
    baseline_header = ["task_id", "baseline_passed", "baseline_total", "baseline_acc_pct", "has_failing_test", "target_idx", "note"]

    attempts_fp, attempts_writer = _open_tsv_for_resume(attempts_tsv, attempts_header, force=False)
    baseline_fp, baseline_writer = _open_tsv_for_resume(baseline_tsv, baseline_header, force=False)

    nb_passed = int(resume_state["nb_passed"])
    nb_total = int(resume_state["nb_total"])
    nb_target = int(resume_state["nb_target"])
    bt_passed = int(resume_state["bt_passed"])
    bt_total = int(resume_state["bt_total"])
    bt_target = int(resume_state["bt_target"])
    nb_counts: Dict[str, int] = dict(resume_state["nb_counts"])
    bt_counts: Dict[str, int] = dict(resume_state["bt_counts"])
    evaluated = int(resume_state["evaluated"])
    skipped = int(resume_state["skipped"])
    pending: List[Dict[str, Any]] = []

    def write_result(item: Dict[str, Any], result: Dict[str, Any]) -> None:
        nonlocal nb_passed, nb_total, nb_target, bt_passed, bt_total, bt_target, evaluated
        evaluated += 1
        nb_eval = result["nonbacktracking_eval"]
        bt_eval = result["backtracking_eval"]
        nb_passed += int(nb_eval["full_suite_passed"]); nb_total += int(nb_eval["full_suite_total"]); nb_target += int(bool(nb_eval.get("target_ok", False)))
        bt_passed += int(bt_eval["full_suite_passed"]); bt_total += int(bt_eval["full_suite_total"]); bt_target += int(bool(bt_eval.get("target_ok", False)))
        _add_count(nb_counts, result["selected_source"]); _add_count(bt_counts, result["backtracking_source"])
        debug_eval_row = result.get("debug_eval") or {}
        generation_eval_row = result.get("generation_eval") or {}
        debug_verifier_eval_row = result.get("debug_verifier_eval") or {}
        generation_verifier_eval_row = result.get("generation_verifier_eval") or {}

        attempts_writer.writerow([
            experiment.name, item["task_id"], item["target_idx"],
            item["baseline_eval"]["full_suite_passed"], item["baseline_eval"]["full_suite_total"], f"{item['baseline_eval']['full_suite_acc_pct']:.2f}",
            debug_eval_row.get("full_suite_passed", ""), debug_eval_row.get("full_suite_total", ""), f"{debug_eval_row.get('full_suite_acc_pct', 0.0):.2f}" if debug_eval_row else "",
            generation_eval_row.get("full_suite_passed", ""), generation_eval_row.get("full_suite_total", ""), f"{generation_eval_row.get('full_suite_acc_pct', 0.0):.2f}" if generation_eval_row else "",
            debug_verifier_eval_row.get("full_suite_passed", ""), debug_verifier_eval_row.get("full_suite_total", ""),
            generation_verifier_eval_row.get("full_suite_passed", ""), generation_verifier_eval_row.get("full_suite_total", ""),
            result["selected_source"], nb_eval["full_suite_passed"], nb_eval["full_suite_total"], f"{nb_eval['full_suite_acc_pct']:.2f}",
            result["backtracking_source"], bt_eval["full_suite_passed"], bt_eval["full_suite_total"], f"{bt_eval['full_suite_acc_pct']:.2f}",
            int(bool(nb_eval.get("target_ok", False))), result.get("debug_prompt_chars", 0), result.get("generation_prompt_chars", 0), result.get("debug_output_chars", 0), result.get("generation_output_chars", 0),
            repr(item.get("target_input")), repr(item.get("target_expected")), repr(item.get("target_got_baseline")), repr(nb_eval.get("target_got")),
            result.get("note", ""), result.get("debug_code", ""), result.get("generation_code", ""), result.get("selected_code", ""),
        ])
        attempts_fp.flush()
        save_jsonl_line(summary_jsonl, {
            "event": "final_metrics",
            "experiment": experiment.name,
            "task_id": item["task_id"],
            "target_idx": item["target_idx"],
            "baseline_eval": {k: v for k, v in item["baseline_eval"].items() if k not in ("status", "exec_out")},
            "selected_source": result["selected_source"],
            "nonbacktracking_eval": {k: v for k, v in nb_eval.items() if k not in ("status", "exec_out", "results")},
            "backtracking_source": result["backtracking_source"],
            "backtracking_eval": {k: v for k, v in bt_eval.items() if k not in ("status", "exec_out", "results")},
            "debug_eval": result.get("debug_eval"),
            "generation_eval": result.get("generation_eval"),
            "debug_verifier_eval": result.get("debug_verifier_eval"),
            "generation_verifier_eval": result.get("generation_verifier_eval"),
            "candidate_evals": result.get("candidate_evals"),
            "candidate_codes": result.get("candidate_codes"),
            "selected_sample_index": result.get("selected_sample_index"),
            "note": result.get("note", ""),
        })
        completed_ids.add(item["task_id"])

    def write_skipped_prepare(task_id: str, note: str) -> None:
        nonlocal skipped
        skipped += 1
        tid = str(task_id or "").strip() or f"__missing_task_id_{skipped}"
        save_jsonl_line(summary_jsonl, {
            "event": "skipped_prepare_row",
            "experiment": experiment.name,
            "task_id": tid,
            "note": note,
        })
        completed_ids.add(tid)

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        need_debug = experiment.kind in ("single_debug", "best2_debug_oracle", "best2_debug_verifier", "routing_oracle", "routing_verifier")
        need_gen = experiment.kind in ("single_generation", "best2_generation_oracle", "best2_generation_verifier", "routing_oracle", "routing_verifier")
        best2_debug = experiment.kind in ("best2_debug_oracle", "best2_debug_verifier")
        best2_gen = experiment.kind in ("best2_generation_oracle", "best2_generation_verifier")
        debug_outs: List[Tuple[str, int]] = []
        gen_outs: List[Tuple[str, int]] = []
        if need_debug:
            debug_runner = runners[experiment.debug_model]
            if best2_debug:
                debug_msgs = []
                for x in pending:
                    msg = build_debug_messages(x["buggy_code"], x["target_input"], x["target_expected"], x["target_got_baseline"], x["entry_point"])
                    debug_msgs.extend([msg, msg])
            else:
                debug_msgs = [build_debug_messages(x["buggy_code"], x["target_input"], x["target_expected"], x["target_got_baseline"], x["entry_point"]) for x in pending]
            debug_outs = debug_runner.generate_batch(debug_msgs, args.deterministic_test, args.max_new_tokens, args.temperature, args.top_p)
        if need_gen:
            gen_runner = runners[experiment.generation_model]
            if best2_gen:
                gen_msgs = []
                for x in pending:
                    msg = build_generation_messages(x["prompt"], x["signature"], x["entry_point"])
                    gen_msgs.extend([msg, msg])
            else:
                gen_msgs = [build_generation_messages(x["prompt"], x["signature"], x["entry_point"]) for x in pending]
            gen_outs = gen_runner.generate_batch(gen_msgs, args.deterministic_test, args.max_new_tokens, args.temperature, args.top_p)

        for i, item in enumerate(pending):
            if best2_debug:
                d_slice = debug_outs[2 * i: 2 * i + 2]
                debug_raws = [x[0] for x in d_slice]
                debug_prompt_len = d_slice[0][1] if d_slice else 0
            else:
                debug_raws = [debug_outs[i][0]] if need_debug else []
                debug_prompt_len = debug_outs[i][1] if need_debug else 0
            if best2_gen:
                g_slice = gen_outs[2 * i: 2 * i + 2]
                gen_raws = [x[0] for x in g_slice]
                gen_prompt_len = g_slice[0][1] if g_slice else 0
            else:
                gen_raws = [gen_outs[i][0]] if need_gen else []
                gen_prompt_len = gen_outs[i][1] if need_gen else 0

            debug_codes = [extract_fenced_code(x) for x in debug_raws]
            generation_codes = [extract_generation_code(x) for x in gen_raws]
            debug_raw = debug_raws[0] if debug_raws else ""
            gen_raw = gen_raws[0] if gen_raws else ""
            debug_code = debug_codes[0] if debug_codes else ""
            generation_code = generation_codes[0] if generation_codes else ""

            debug_eval = evaluate_oracle_candidate(item["entry_point"], debug_code, item["unit_inputs"], item["unit_outputs"], item["use_set"], args.timeout, item["target_idx"]) if need_debug and not best2_debug else None
            generation_eval = evaluate_oracle_candidate(item["entry_point"], generation_code, item["unit_inputs"], item["unit_outputs"], item["use_set"], args.timeout, item["target_idx"]) if need_gen and not best2_gen else None
            selected_source = "buggy"
            selected_code = item["buggy_code"]
            selected_eval = item["baseline_eval"]
            debug_verifier_eval = None
            generation_verifier_eval = None
            note = item.get("note", "")
            candidate_evals = None
            candidate_codes = None
            selected_sample_index = None

            if experiment.kind == "single_debug":
                selected_source, selected_code, selected_eval = "debug", debug_code, debug_eval
            elif experiment.kind == "single_generation":
                selected_source, selected_code, selected_eval = "generation", generation_code, generation_eval
            elif experiment.kind == "best2_generation_oracle":
                generation_evals = [evaluate_oracle_candidate(item["entry_point"], code, item["unit_inputs"], item["unit_outputs"], item["use_set"], args.timeout, item["target_idx"]) for code in generation_codes]
                selected_sample_index = choose_best_oracle_candidate(generation_evals)
                selected_source = f"generation_sample_{selected_sample_index + 1}"
                selected_code = generation_codes[selected_sample_index]
                selected_eval = generation_evals[selected_sample_index]
                generation_eval = selected_eval
                generation_code = selected_code
                gen_raw = gen_raws[selected_sample_index]
                candidate_evals = [public_eval(ev) for ev in generation_evals]
                candidate_codes = generation_codes
                note = (note + ";" if note else "") + summarize_candidate_evals("gen", generation_evals, selected_sample_index)
            elif experiment.kind == "best2_debug_oracle":
                debug_evals = [evaluate_oracle_candidate(item["entry_point"], code, item["unit_inputs"], item["unit_outputs"], item["use_set"], args.timeout, item["target_idx"]) for code in debug_codes]
                selected_sample_index = choose_best_oracle_candidate(debug_evals)
                selected_source = f"debug_sample_{selected_sample_index + 1}"
                selected_code = debug_codes[selected_sample_index]
                selected_eval = debug_evals[selected_sample_index]
                debug_eval = selected_eval
                debug_code = selected_code
                debug_raw = debug_raws[selected_sample_index]
                candidate_evals = [public_eval(ev) for ev in debug_evals]
                candidate_codes = debug_codes
                note = (note + ";" if note else "") + summarize_candidate_evals("debug", debug_evals, selected_sample_index)

            elif experiment.kind == "best2_generation_verifier":
                verifier_test_code = item["verifier_test_code"]
                generation_verifier_evals = [
                    evaluate_verifier_candidate(item["entry_point"], code, verifier_test_code, args.timeout, 0)
                    for code in generation_codes
                ]
                selected_sample_index = choose_best_verifier_candidate(generation_verifier_evals)
                selected_source = f"generation_sample_{selected_sample_index + 1}"
                selected_code = generation_codes[selected_sample_index]
                selected_eval = evaluate_oracle_candidate(
                    item["entry_point"], selected_code,
                    item["unit_inputs"], item["unit_outputs"], item["use_set"],
                    args.timeout, item["target_idx"]
                )
                generation_eval = selected_eval
                generation_code = selected_code
                gen_raw = gen_raws[selected_sample_index]
                generation_verifier_eval = generation_verifier_evals[selected_sample_index]
                candidate_evals = [{"verifier_eval": {k: v for k, v in ev.items() if k != "results"}} for ev in generation_verifier_evals]
                candidate_codes = generation_codes
                note = (note + ";" if note else "") + summarize_candidate_evals("verifier_gen", generation_verifier_evals, selected_sample_index)

            elif experiment.kind == "best2_debug_verifier":
                verifier_test_code = item["verifier_test_code"]
                debug_verifier_evals = [
                    evaluate_verifier_candidate(item["entry_point"], code, verifier_test_code, args.timeout, 0)
                    for code in debug_codes
                ]
                selected_sample_index = choose_best_verifier_candidate(debug_verifier_evals)
                selected_source = f"debug_sample_{selected_sample_index + 1}"
                selected_code = debug_codes[selected_sample_index]
                selected_eval = evaluate_oracle_candidate(
                    item["entry_point"], selected_code,
                    item["unit_inputs"], item["unit_outputs"], item["use_set"],
                    args.timeout, item["target_idx"]
                )
                debug_eval = selected_eval
                debug_code = selected_code
                debug_raw = debug_raws[selected_sample_index]
                debug_verifier_eval = debug_verifier_evals[selected_sample_index]
                candidate_evals = [{"verifier_eval": {k: v for k, v in ev.items() if k != "results"}} for ev in debug_verifier_evals]
                candidate_codes = debug_codes
                note = (note + ";" if note else "") + summarize_candidate_evals("verifier_debug", debug_verifier_evals, selected_sample_index)

            elif experiment.kind == "routing_oracle":
                selected_source = choose_candidate(debug_eval, generation_eval)
                if selected_source == "debug":
                    selected_code, selected_eval = debug_code, debug_eval
                else:
                    selected_code, selected_eval = generation_code, generation_eval
            elif experiment.kind == "routing_verifier":
                verifier_test_code = item["verifier_test_code"]
                debug_verifier_eval = evaluate_verifier_candidate(item["entry_point"], debug_code, verifier_test_code, args.timeout, 0)
                generation_verifier_eval = evaluate_verifier_candidate(item["entry_point"], generation_code, verifier_test_code, args.timeout, 0)
                selected_source = choose_candidate(debug_verifier_eval, generation_verifier_eval)
                selected_code = debug_code if selected_source == "debug" else generation_code
                selected_eval = evaluate_oracle_candidate(item["entry_point"], selected_code, item["unit_inputs"], item["unit_outputs"], item["use_set"], args.timeout, item["target_idx"])

            if selected_eval["full_suite_passed"] > item["baseline_eval"]["full_suite_passed"]:
                backtracking_source = selected_source
                bt_eval = selected_eval
            else:
                backtracking_source = "buggy"
                bt_eval = item["baseline_eval"]

            write_result(item, {
                "selected_source": selected_source,
                "selected_code": selected_code,
                "nonbacktracking_eval": selected_eval,
                "backtracking_source": backtracking_source,
                "backtracking_eval": bt_eval,
                "debug_eval": {k: v for k, v in debug_eval.items() if k not in ("status", "exec_out")} if debug_eval else None,
                "generation_eval": {k: v for k, v in generation_eval.items() if k not in ("status", "exec_out")} if generation_eval else None,
                "debug_verifier_eval": {k: v for k, v in debug_verifier_eval.items() if k != "results"} if debug_verifier_eval else None,
                "generation_verifier_eval": {k: v for k, v in generation_verifier_eval.items() if k != "results"} if generation_verifier_eval else None,
                "candidate_evals": candidate_evals,
                "candidate_codes": candidate_codes,
                "selected_sample_index": selected_sample_index,
                "debug_code": debug_code,
                "generation_code": generation_code,
                "debug_prompt_chars": debug_prompt_len,
                "generation_prompt_chars": gen_prompt_len,
                "debug_output_chars": len(debug_raw),
                "generation_output_chars": len(gen_raw),
                "note": note,
            })
        pending = []

    try:
        for ex in tqdm(rows, desc=f"[{experiment.name}]", mininterval=args.tqdm_mininterval, disable=args.disable_tqdm):
            task_id = str(ex.get("task_id", "")).strip()
            if task_id and task_id in completed_ids:
                continue
            item = prepare_row(ex, args)
            if item is None:
                write_skipped_prepare(task_id, "skipped_prepare_row_no_tests_or_missing_fields")
                continue
            if item["task_id"] in completed_ids:
                continue
            has_failing = item["target_idx"] >= 0
            baseline_writer.writerow([
                item["task_id"], item["baseline_eval"]["full_suite_passed"], item["baseline_eval"]["full_suite_total"], f"{item['baseline_eval']['full_suite_acc_pct']:.2f}",
                int(has_failing), item["target_idx"], "" if has_failing else "no_failing_oracle_test",
            ])
            baseline_fp.flush()

            if experiment.kind == "buggy":
                write_result(item, {
                    "selected_source": "buggy",
                    "selected_code": item["buggy_code"],
                    "nonbacktracking_eval": item["baseline_eval"],
                    "backtracking_source": "buggy",
                    "backtracking_eval": item["baseline_eval"],
                    "note": "baseline_only",
                })
                continue

            if not has_failing and experiment.kind in ("single_debug", "best2_debug_oracle", "best2_debug_verifier", "routing_oracle", "routing_verifier"):
                # Debug prompt needs a failing target. For generation-only, still allow generation.
                write_result(item, {
                    "selected_source": "buggy",
                    "selected_code": item["buggy_code"],
                    "nonbacktracking_eval": item["baseline_eval"],
                    "backtracking_source": "buggy",
                    "backtracking_eval": item["baseline_eval"],
                    "note": "skipped_no_failing_oracle_test",
                })
                continue

            if experiment.kind in ("routing_verifier", "best2_generation_verifier", "best2_debug_verifier"):
                vmap = verifier_maps.get(experiment.verifier_source or "", {})
                vinfo = vmap.get(item["task_id"])
                if vinfo is None or not str(vinfo.get("verifier_test_code", "")).strip():
                    write_result(item, {
                        "selected_source": "buggy",
                        "selected_code": item["buggy_code"],
                        "nonbacktracking_eval": item["baseline_eval"],
                        "backtracking_source": "buggy",
                        "backtracking_eval": item["baseline_eval"],
                        "note": "skipped_missing_or_empty_verifier",
                    })
                    continue
                item["verifier_test_code"] = str(vinfo["verifier_test_code"])
                item["verifier_num_asserts"] = int(vinfo.get("num_asserts", 0))

            pending.append(item)
            if len(pending) >= args.batch_size:
                flush_pending()
        flush_pending()
    finally:
        attempts_fp.close(); baseline_fp.close()

    problems_seen = len(rows)
    final_non = build_final_metrics(experiment, "non_backtracking", problems_seen, evaluated, skipped, nb_passed, nb_total, nb_target, nb_counts, args)
    final_bt = build_final_metrics(experiment, "backtracking", problems_seen, evaluated, skipped, bt_passed, bt_total, bt_target, bt_counts, args)
    with open(final_non_path, "w", encoding="utf-8") as fp:
        json.dump(final_non, fp, ensure_ascii=False, indent=2)
    with open(final_bt_path, "w", encoding="utf-8") as fp:
        json.dump(final_bt, fp, ensure_ascii=False, indent=2)
    shutil.copy2(attempts_tsv, os.path.join(out_bt, os.path.basename(attempts_tsv)))
    shutil.copy2(baseline_tsv, os.path.join(out_bt, os.path.basename(baseline_tsv)))
    shutil.copy2(summary_jsonl, os.path.join(out_bt, os.path.basename(summary_jsonl)))

    final_all = {"experiment": experiment.name, "table_label": experiment.table_label, "non_backtracking": final_non, "backtracking": final_bt}
    with open(final_all_path, "w", encoding="utf-8") as fp:
        json.dump(final_all, fp, ensure_ascii=False, indent=2)
    return final_all

def needed_models(experiments: List[Experiment]) -> Tuple[bool, bool]:
    need_base = any(e.debug_model == "base" or e.generation_model == "base" for e in experiments)
    need_rl = any(e.debug_model == "rl" or e.generation_model == "rl" for e in experiments)
    return need_base, need_rl


def write_combined_summary(log_dir: str, results: List[Dict[str, Any]]) -> None:
    ensure_dir(log_dir)
    with open(os.path.join(log_dir, "all_selected_experiments_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    tsv = os.path.join(log_dir, "all_selected_experiments_summary.tsv")
    with open(tsv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerow([
            "experiment", "table_label", "mode", "accuracy_pct", "passed", "total", "evaluated",
            "generation_count", "debug_count", "buggy_count", "source_counts",
        ])
        for r in results:
            for mode_key in ("non_backtracking", "backtracking"):
                m = r[mode_key]
                w.writerow([
                    r["experiment"], r["table_label"], m["mode"], f"{m['overall_accuracy_pct']:.4f}",
                    m["final_full_suite_passed_total"], m["final_full_suite_total_asserts"], m["evaluated"],
                    m.get("generation_count", 0), m.get("debug_count", 0), m.get("buggy_count", 0),
                    json.dumps(m.get("source_counts", {}), sort_keys=True),
                ])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HE+Fix runner for the nine Qwen rows in the PATSSEL main table.")
    ap.add_argument("--log_dir", type=str, required=True)
    ap.add_argument("--all", action="store_true", help="Run all HE+Fix table experiments.")
    ap.add_argument("--experiments", nargs="*", default=[], choices=sorted(EXPERIMENTS.keys()))
    ap.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    ap.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    ap.add_argument("--limit_tasks", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target_policy", choices=["first", "random"], default="first")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)

    ap.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL_ID)
    ap.add_argument("--rl_peft_checkpoint", type=str, default=DEFAULT_RL_PEFT_CHECKPOINT)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--deterministic_test", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    ap.add_argument("--base_testgen_jsonl", type=str, default=DEFAULT_BASE_TESTGEN_JSONL)
    ap.add_argument("--rl_testgen_jsonl", type=str, default=DEFAULT_RL_TESTGEN_JSONL)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--disable_tqdm", action="store_true")
    ap.add_argument("--tqdm_mininterval", type=float, default=5.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.log_dir = expand_path(args.log_dir)
    args.rl_peft_checkpoint = expand_path(args.rl_peft_checkpoint)
    args.base_testgen_jsonl = expand_path(args.base_testgen_jsonl)
    args.rl_testgen_jsonl = expand_path(args.rl_testgen_jsonl)
    ensure_dir(args.log_dir)
    set_global_seeds(args.seed)

    if args.all:
        exp_names = ALL_EXPERIMENT_ORDER
    else:
        if not args.experiments:
            raise ValueError("Pass --all or at least one --experiments value.")
        exp_names = args.experiments
    experiments = [EXPERIMENTS[name] for name in exp_names]

    need_base, need_rl = needed_models(experiments)
    need_base_verifier = any(e.verifier_source == "base_testgen" for e in experiments)
    need_rl_verifier = any(e.verifier_source == "rl_testgen" for e in experiments)

    print(f"[dataset] Loading {args.dataset_name} split={args.split}")
    rows = load_hefix_rows(args)
    print(f"[dataset] rows={len(rows)}")

    verifier_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if need_base_verifier:
        print(f"[verifier] Loading base testgen verifier: {args.base_testgen_jsonl}")
        verifier_maps["base_testgen"] = load_verifier_tests_map(args.base_testgen_jsonl)
    if need_rl_verifier:
        print(f"[verifier] Loading RL testgen verifier: {args.rl_testgen_jsonl}")
        verifier_maps["rl_testgen"] = load_verifier_tests_map(args.rl_testgen_jsonl)

    runners: Dict[str, ModelRunner] = {}
    try:
        if need_base:
            print(f"[model] Loading base model: {args.base_model_id}")
            runners["base"] = load_model_and_tokenizer(args.base_model_id, peft_checkpoint=None, dtype=args.dtype)
        if need_rl:
            if not args.rl_peft_checkpoint or not os.path.isdir(args.rl_peft_checkpoint):
                raise FileNotFoundError(f"RL PEFT checkpoint not found: {args.rl_peft_checkpoint}")
            print(f"[model] Loading RL PEFT debugger/generator: base={args.base_model_id}, adapter={args.rl_peft_checkpoint}")
            runners["rl"] = load_model_and_tokenizer(args.base_model_id, peft_checkpoint=args.rl_peft_checkpoint, dtype=args.dtype)

        all_results: List[Dict[str, Any]] = []
        for exp in experiments:
            print(f"\n===== Running {exp.name}: {exp.table_label} =====")
            res = run_experiment(exp, args, rows, runners, verifier_maps)
            all_results.append(res)
            write_combined_summary(args.log_dir, all_results)

        print("\n=== HE+Fix table experiments complete ===")
        print(f"Summary: {os.path.join(args.log_dir, 'all_selected_experiments_summary.tsv')}")
    finally:
        for r in runners.values():
            free_runner(r)


if __name__ == "__main__":
    main()
