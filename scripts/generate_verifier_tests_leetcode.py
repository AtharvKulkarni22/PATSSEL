#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import logging
import multiprocessing as mp
import random
import re
import textwrap
import traceback
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm.auto import tqdm

SYSTEM_PROMPT = """You are a Python unit-test generation assistant for programming problems.
Your job is to generate executable Python tests in exactly the dataset style.

REQUIRED OUTPUT FORMAT:
- Output ONLY one Python code block.
- Inside the code block, define exactly one function:

  def check(candidate):
      assert candidate(...) == ...
      assert candidate(...) == ...
      ...

Rules:
- Use the provided entry point implicitly through `candidate`.
- Do not include explanations.
- Do not include imports.
- Do not include any text outside the code block.
- Generate as many tests as needed to aim for 100% code coverage of the provided correct solution.
- Include diverse asserts: easy, edge, boundary, duplicate, large-value, adversarial, empty/singleton, and structure-specific cases when applicable.
- Every assert must be deterministic.
- The expected output must be correct with respect to the provided correct solution.
- If no valid output exists for an input, you may use `None` when consistent with the provided correct solution behavior.
"""

PY_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
CHECK_HEADER_RE = re.compile(r"^\s*def\s+check\s*\(\s*candidate\s*\)\s*:\s*$")
ASSERT_START_RE = re.compile(r"^\s*assert\b")
SAFE_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
DISALLOWED_HELPER_PREFIXES = (
    "import ", "from ", "def ", "class ", "for ", "while ", "if ", "elif ", "else:",
    "try:", "except", "finally:", "with ", "return ", "yield ", "@"
)

PROMPT_IMPORTS = """\
import random, functools, collections, string, math, datetime
from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *
from collections import deque

inf = float('inf')

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
            i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
            i += 1
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    elif not p or not q:
        return False
    elif p.val != q.val:
        return False
    else:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
"""


DEFAULT_SEED = 1012


def stable_int_from_str(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def split_df_half_deterministic(df: pd.DataFrame, seed: int, split_name: str, mode: str) -> pd.DataFrame:
    """Apply the historical half split, or keep an already materialized public split."""
    assert mode in ("dev", "test", "all")
    if mode == "all":
        return df.reset_index(drop=True)
    rs = np.random.RandomState(seed + stable_int_from_str(split_name))
    idx = np.arange(len(df))
    rs.shuffle(idx)
    mid = len(idx) // 2
    chosen = idx[mid:] if mode == "test" else idx[:mid]
    return df.iloc[chosen].reset_index(drop=True)


def normalize_whitespace(text: str) -> str:
    return textwrap.dedent(text or "").strip()


def strip_markdown_fences(text: str) -> str:
    s = (text or "").strip()
    matches = PY_FENCE_RE.findall(s)
    if matches:
        return normalize_whitespace(matches[-1])
    if s.startswith("```"):
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return normalize_whitespace("\n".join(lines))
    return normalize_whitespace(s)


def extract_python_block(text: str) -> str:
    return strip_markdown_fences(text)


def looks_like_check_function(text: str) -> bool:
    return bool(re.search(r"def\s+check\s*\(\s*candidate\s*\)\s*:", text or ""))


def make_fewshot_user(example: Dict[str, Any]) -> str:
    return (
        f"Problem description:\n{normalize_whitespace(example['problem_description'])}\n\n"
        f"Starter code:\n{normalize_whitespace(example['starter_code'])}\n\n"
        f"Correct solution:\n{normalize_whitespace(example['completion'])}\n\n"
        f"Entry point:\n{example['entry_point']}\n\n"
        "Generate Python tests in the dataset format."
    )


def make_fewshot_assistant(example: Dict[str, Any]) -> str:
    test_code = normalize_whitespace(example["test"])
    return f"```python\n{test_code}\n```"


def build_prompt(tokenizer, row: Dict[str, Any], fewshot_examples: List[Dict[str, Any]]):
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": make_fewshot_user(ex)})
        messages.append({"role": "assistant", "content": make_fewshot_assistant(ex)})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Problem description:\n{normalize_whitespace(row['problem_description'])}\n\n"
                f"Starter code:\n{normalize_whitespace(row['starter_code'])}\n\n"
                f"Correct solution:\n{normalize_whitespace(row['completion'])}\n\n"
                f"Entry point:\n{row['entry_point']}\n\n"
                "Generate Python tests in the dataset format now. Generate as many tests as needed to aim for 100% code coverage of the correct solution."
            ),
        }
    )
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_tester(path: str):
    spec = importlib.util.spec_from_file_location("tester_mod", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import tester module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = [
        "resolve_entry_point",
        "eval_all_asserts_with_output",
        "eval_single_assert_with_output",
        "extract_assert_texts",
        "TIME_LIMIT_SEC",
    ]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"tester module missing required symbol: {name}")
    return mod


def load_task_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("task_ids.json must be a JSON list")
    return [str(x) for x in data]


def load_buggy_rows(paths: List[str], split_mode: str, seed: int) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
        df["source_tsv"] = p
        df = split_df_half_deterministic(df, seed=seed, split_name=Path(p).name, mode=split_mode)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    for col in ["pass", "fail", "total", "pass_rate", "baseline_total", "baseline_failed", "baseline_passed", "question_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_dataset_rows(dataset_name: str, split_name: str, task_ids: Iterable[str]) -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name)
    if split_name not in ds:
        raise ValueError(f"Requested split '{split_name}' not found. Available splits: {list(ds.keys())}")
    wanted = set(task_ids)
    rows: List[Dict[str, Any]] = []
    for row in ds[split_name]:
        if str(row.get("task_id", "")) in wanted:
            rows.append(row)
    return rows


def choose_fewshot_examples(rows: List[Dict[str, Any]], k: int, exclude_task_id: str) -> List[Dict[str, Any]]:
    examples = [
        r for r in rows
        if str(r.get("task_id")) != str(exclude_task_id) and r.get("test") and r.get("completion")
    ]
    return examples[:k]


def resolve_torch_dtype(dtype: str):
    if dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(
    model_name: str,
    dtype: str,
    peft_model_path: Optional[str] = None,
    merge_peft: bool = False,
):
    """
    Load either:
      1) a normal/full model from --model_name, or
      2) a base model from --model_name plus a PEFT LoRA adapter from --peft_model_path.

    For your checkpoint-400 directory, use:
      --model_name Qwen/Qwen3-4B-Instruct-2507
      --peft_model_path checkpoints/test_generator
    """
    tokenizer_source = peft_model_path if peft_model_path else model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = resolve_torch_dtype(dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if peft_model_path:
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_path,
            is_trainable=False,
        )
        if merge_peft:
            model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    return model, tokenizer


def generate_tests(
    model,
    tokenizer,
    row: Dict[str, Any],
    fewshots: List[Dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    inputs = build_prompt(tokenizer, row, fewshots).to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    test_code = extract_python_block(gen_text)
    return gen_text, test_code


def parse_asserts_from_check(test_code: str) -> List[ast.Assert]:
    tree = ast.parse(normalize_whitespace(test_code))
    out: List[ast.Assert] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            out.append(node)
    out.sort(key=lambda x: (getattr(x, "lineno", 0), getattr(x, "col_offset", 0)))
    return out


def render_assert_source(assert_node: ast.Assert, source: str) -> str:
    seg = ast.get_source_segment(source, assert_node)
    return seg.strip() if seg else ""


def _is_safe_expr(expr_src: str) -> bool:
    """
    Allow only simple expressions for helper assignments.
    We intentionally exclude comprehensions, lambdas, calls to arbitrary functions, etc.
    """
    try:
        tree = ast.parse(expr_src, mode="eval")
    except Exception:
        return False

    allowed_call_names = {
        "list_node", "tree_node", "sorted", "tuple", "list", "set", "dict",
        "len", "sum", "min", "max", "abs", "range"
    }

    class Visitor(ast.NodeVisitor):
        ok = True

        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_call_names:
                for arg in node.args:
                    self.visit(arg)
                for kw in node.keywords:
                    self.visit(kw.value)
                return
            self.ok = False

        def visit_Attribute(self, node: ast.Attribute):
            self.ok = False

        def visit_Lambda(self, node: ast.Lambda):
            self.ok = False

        def visit_ListComp(self, node: ast.ListComp):
            self.ok = False

        def visit_SetComp(self, node: ast.SetComp):
            self.ok = False

        def visit_DictComp(self, node: ast.DictComp):
            self.ok = False

        def visit_GeneratorExp(self, node: ast.GeneratorExp):
            self.ok = False

        def generic_visit(self, node):
            allowed = (
                ast.Expression, ast.Constant, ast.Name, ast.Load,
                ast.List, ast.Tuple, ast.Set, ast.Dict,
                ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.Invert,
                ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
                ast.BoolOp, ast.And, ast.Or,
                ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
                ast.Subscript, ast.Slice, ast.Index,
                ast.Call,
            )
            if not isinstance(node, allowed):
                self.ok = False
                return
            super().generic_visit(node)

    v = Visitor()
    v.visit(tree)
    return bool(v.ok)


def _parse_single_stmt(stmt_src: str) -> Optional[ast.stmt]:
    try:
        tree = ast.parse(stmt_src)
        if len(tree.body) != 1:
            return None
        return tree.body[0]
    except Exception:
        return None


def _is_safe_helper_stmt(stmt_src: str) -> bool:
    stripped = stmt_src.strip()
    if not stripped:
        return False
    low = stripped.lower()
    if any(low.startswith(prefix) for prefix in DISALLOWED_HELPER_PREFIXES):
        return False
    if CHECK_HEADER_RE.match(stripped):
        return False
    if stripped.startswith("#"):
        return False

    stmt = _parse_single_stmt(stripped)
    if stmt is None:
        return False
    if not isinstance(stmt, ast.Assign):
        return False
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return False

    m = SAFE_ASSIGN_RE.match(stripped)
    if not m:
        return False
    rhs = m.group(2)
    return _is_safe_expr(rhs)


def _count_balance(text: str) -> int:
    score = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}
    in_str = False
    quote = ""
    esc = False

    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in {"'", '"'}:
            in_str = True
            quote = ch
            continue
        if ch in pairs:
            score += 1
        elif ch in closing:
            score -= 1
    return score


def salvage_structured_blocks(text: str) -> Dict[str, Any]:
    """
    Recover:
    - single-line asserts
    - multi-line asserts
    - safe helper assignments

    Build a normalized:
        def check(candidate):
            <helpers>
            <asserts>
    """
    src = strip_markdown_fences(text)
    lines = src.splitlines()

    helpers: List[str] = []
    asserts: List[str] = []
    dropped: List[str] = []
    seen_helpers: Set[str] = set()
    seen_asserts: Set[str] = set()

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        stripped = raw.strip()

        if not stripped:
            i += 1
            continue
        if stripped.startswith("#"):
            i += 1
            continue
        if CHECK_HEADER_RE.match(stripped):
            i += 1
            continue
        if stripped.startswith("```"):
            i += 1
            continue

        # Multi-line or single-line assert salvage.
        if ASSERT_START_RE.match(stripped):
            block_lines = [stripped]
            j = i + 1

            # Grow block while unbalanced or until parsing succeeds.
            while True:
                joined = "\n".join(block_lines)
                stmt = _parse_single_stmt(joined)
                if isinstance(stmt, ast.Assert):
                    if joined not in seen_asserts:
                        seen_asserts.add(joined)
                        asserts.append(joined)
                    i = j
                    break

                if j >= len(lines):
                    # Could not salvage this assert block.
                    dropped.append(joined)
                    i = j
                    break

                nxt = lines[j].rstrip("\n")
                nxt_stripped = nxt.strip()

                # Stop if a brand-new top-level assert/check/helper likely begins and current block is hopeless.
                if (ASSERT_START_RE.match(nxt_stripped) or CHECK_HEADER_RE.match(nxt_stripped)) and _count_balance(joined) <= 0:
                    dropped.append(joined)
                    i = j
                    break

                if nxt_stripped.startswith("```"):
                    dropped.append(joined)
                    i = j + 1
                    break

                block_lines.append(nxt_stripped)
                j += 1
            continue

        # Safe helper salvage.
        if _is_safe_helper_stmt(stripped):
            if stripped not in seen_helpers:
                seen_helpers.add(stripped)
                helpers.append(stripped)
            i += 1
            continue

        # Try trimming prose/comment suffix from an otherwise-valid helper line.
        if "  #" in stripped:
            head = stripped.split("  #", 1)[0].rstrip()
            if _is_safe_helper_stmt(head):
                if head not in seen_helpers:
                    seen_helpers.add(head)
                    helpers.append(head)
                i += 1
                continue

        dropped.append(stripped)
        i += 1

    clean_lines: List[str] = ["def check(candidate):"]
    if not helpers and not asserts:
        clean_lines.append("    pass")
    else:
        for h in helpers:
            clean_lines.append(f"    {h}")
        for a in asserts:
            indented = "\n".join("    " + ln for ln in a.splitlines())
            clean_lines.append(indented)

    clean_test_code = "\n".join(clean_lines)
    parse_error = ""
    try:
        ast.parse(clean_test_code)
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"

    return {
        "clean_test_code": clean_test_code,
        "helper_lines": helpers,
        "assert_blocks": asserts,
        "num_helper_lines": len(helpers),
        "num_assert_blocks": len(asserts),
        "dropped_lines": dropped,
        "parse_error": parse_error,
    }


def salvage_clean_test_code(raw_generation: str, extracted_test_code: str) -> Dict[str, Any]:
    # Prefer structured extraction from extracted block, then supplement from full raw output.
    first = salvage_structured_blocks(extracted_test_code)
    second = salvage_structured_blocks(raw_generation)

    merged_helpers: List[str] = []
    merged_asserts: List[str] = []
    seen_h: Set[str] = set()
    seen_a: Set[str] = set()

    for item in first["helper_lines"] + second["helper_lines"]:
        if item not in seen_h:
            seen_h.add(item)
            merged_helpers.append(item)

    for item in first["assert_blocks"] + second["assert_blocks"]:
        if item not in seen_a:
            seen_a.add(item)
            merged_asserts.append(item)

    clean_lines: List[str] = ["def check(candidate):"]
    if not merged_helpers and not merged_asserts:
        clean_lines.append("    pass")
    else:
        for h in merged_helpers:
            clean_lines.append(f"    {h}")
        for a in merged_asserts:
            clean_lines.extend("    " + ln for ln in a.splitlines())

    clean_test_code = "\n".join(clean_lines)
    parse_error = ""
    try:
        ast.parse(clean_test_code)
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"

    return {
        "clean_test_code": clean_test_code,
        "salvaged_helper_lines": merged_helpers,
        "salvaged_assert_blocks": merged_asserts,
        "num_salvaged_helper_lines": len(merged_helpers),
        "num_salvaged_assert_blocks": len(merged_asserts),
        "dropped_lines": first["dropped_lines"] + second["dropped_lines"],
        "parse_error": parse_error,
        "salvage_mode": (
            "full_parse"
            if not parse_error and looks_like_check_function(extracted_test_code)
            else "structured_salvage"
        ),
    }


def check_assert_correctness_against_gold(
    tester,
    prompt_text: str,
    gold_code: str,
    test_code: str,
    entry_point: str,
) -> Dict[str, Any]:
    src = normalize_whitespace(test_code)
    try:
        asserts = parse_asserts_from_check(src)
    except Exception as e:
        return {
            "wellformed": False,
            "parse_error": f"{type(e).__name__}: {e}",
            "num_asserts": 0,
            "correct_asserts": 0,
            "incorrect_asserts": 0,
            "assert_details": [],
        }

    details = []
    correct = 0
    incorrect = 0
    for idx, a in enumerate(asserts):
        assert_src = render_assert_source(a, src)
        res = tester.eval_single_assert_with_output(prompt_text, gold_code, src, entry_point, idx)
        is_correct = bool(res.get("ok", False))
        correct += int(is_correct)
        incorrect += int(not is_correct)
        details.append(
            {
                "index": idx,
                "assert_src": assert_src,
                "is_correct_wrt_gold": is_correct,
                "error_type": res.get("error_type", ""),
                "error_msg": res.get("error_msg", ""),
                "stdout": res.get("stdout", ""),
            }
        )
    return {
        "wellformed": True,
        "parse_error": "",
        "num_asserts": len(asserts),
        "correct_asserts": correct,
        "incorrect_asserts": incorrect,
        "assert_details": details,
    }


def _coverage_worker(q, candidate_code: str, test_code: str, entry_point: str, timeout: int):
    try:
        import trace

        candidate_code = normalize_whitespace(candidate_code)
        test_code = normalize_whitespace(test_code)
        traced = trace.Trace(count=True, trace=False)

        full = f"""{PROMPT_IMPORTS}
{candidate_code}
candidate = {entry_point}
{test_code}
check(candidate)
"""
        glb: Dict[str, Any] = {"__name__": "__main__"}
        with contextlib.redirect_stdout(io.StringIO()):
            traced.runctx(full, glb, glb)

        raw_counts = traced.results().counts
        executed_lines = set()
        for (filename, lineno), cnt in raw_counts.items():
            if filename == "<string>" and lineno > 0:
                executed_lines.add(int(lineno))

        prefix_lines = len(PROMPT_IMPORTS.splitlines())
        candidate_line_count = len(candidate_code.splitlines())
        candidate_start = prefix_lines + 1
        candidate_end = candidate_start + candidate_line_count - 1
        candidate_executed = sorted(
            ln - candidate_start + 1
            for ln in executed_lines
            if candidate_start <= ln <= candidate_end
        )
        total_lines = sum(1 for line in candidate_code.splitlines() if line.strip())
        coverage_pct = 100.0 * len(candidate_executed) / max(1, total_lines)
        q.put(
            {
                "ok": True,
                "coverage_pct": coverage_pct,
                "executed_candidate_lines": candidate_executed,
                "total_candidate_nonempty_lines": total_lines,
                "error": "",
            }
        )
    except Exception as e:
        q.put(
            {
                "ok": False,
                "coverage_pct": 0.0,
                "executed_candidate_lines": [],
                "total_candidate_nonempty_lines": 0,
                "error": f"{type(e).__name__}: {e}",
            }
        )


def compute_coverage(candidate_code: str, test_code: str, entry_point: str, timeout: int) -> Dict[str, Any]:
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_coverage_worker, args=(q, candidate_code, test_code, entry_point, timeout))
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join(1)
        return {
            "ok": False,
            "coverage_pct": 0.0,
            "executed_candidate_lines": [],
            "total_candidate_nonempty_lines": 0,
            "error": f"timeout >{timeout}s",
        }
    try:
        return q.get_nowait()
    except Exception:
        return {
            "ok": False,
            "coverage_pct": 0.0,
            "executed_candidate_lines": [],
            "total_candidate_nonempty_lines": 0,
            "error": "no result",
        }


def flush_outputs(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    task_ids: List[str],
) -> None:
    summary_df = pd.DataFrame(summary_rows)
    summary_tsv = output_dir / "summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)

    bug_err = summary_df["buggy_pass_rate_error"].dropna() if not summary_df.empty and "buggy_pass_rate_error" in summary_df.columns else pd.Series(dtype=float)
    agg = {
        "model_name": args.model_name,
        "peft_model_path": args.peft_model_path,
        "merge_peft": args.merge_peft,
        "buggy_subset_split_mode": args.buggy_subset_split_mode,
        "subset_split_seed": args.subset_split_seed,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_requested_task_ids": len(task_ids),
        "num_buggy_rows_evaluated": int(len(summary_df)),
        "num_wellformed_generations": int(summary_df["generated_wellformed"].sum()) if not summary_df.empty else 0,
        "wellformed_rate": float(summary_df["generated_wellformed"].mean()) if not summary_df.empty else 0.0,
        "avg_num_generated_asserts": float(summary_df["num_generated_asserts"].mean()) if not summary_df.empty else 0.0,
        "avg_correct_generated_asserts": float(summary_df["correct_generated_asserts"].mean()) if not summary_df.empty else 0.0,
        "avg_incorrect_generated_asserts": float(summary_df["incorrect_generated_asserts"].mean()) if not summary_df.empty else 0.0,
        "avg_gold_pass_rate_on_generated_tests": float(summary_df["gold_pass_rate_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_buggy_pass_rate_on_generated_tests": float(summary_df["buggy_pass_rate_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_gold_coverage_pct_on_generated_tests": float(summary_df["gold_coverage_pct_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_buggy_coverage_pct_on_generated_tests": float(summary_df["buggy_coverage_pct_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_actual_buggy_pass_rate_from_tsv": float(summary_df["actual_buggy_pass_rate_from_tsv"].dropna().mean()) if not summary_df.empty else 0.0,
        "mean_buggy_pass_rate_error": float(bug_err.mean()) if not bug_err.empty else 0.0,
        "mae_buggy_pass_rate_error": float(bug_err.abs().mean()) if not bug_err.empty else 0.0,
        "files": {
            "per_problem_detailed_jsonl": str(output_dir / "per_problem_detailed.jsonl"),
            "summary_tsv": str(summary_tsv),
        },
        "config": vars(args),
    }

    aggregate_json = output_dir / "aggregate_metrics.json"
    with aggregate_json.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)


def load_completed_keys(detail_jsonl: Path) -> Set[Tuple[str, str, str]]:
    completed: Set[Tuple[str, str, str]] = set()
    if not detail_jsonl.exists():
        return completed
    with detail_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = (str(rec.get("task_id", "")), str(rec.get("question_id", "")), str(rec.get("source_tsv", "")))
            if key[0] and key[1]:
                completed.add(key)
    return completed


def load_existing_summary(summary_tsv: Path) -> List[Dict[str, Any]]:
    if not summary_tsv.exists():
        return []
    try:
        df = pd.read_csv(summary_tsv, sep="\t")
        return df.to_dict(orient="records")
    except Exception:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LeetCode verifier tests with the base or RL-trained Qwen test generator.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Base/full model name or path. When --peft_model_path is set, this must be the base model.")
    parser.add_argument("--peft_model_path", type=str, default=None,
                        help="Optional PEFT LoRA adapter checkpoint path, e.g. checkpoint-400.")
    parser.add_argument("--merge_peft", action="store_true", default=False,
                        help="Merge LoRA adapter into the base model after loading. Usually not needed for inference.")
    parser.add_argument("--task_ids_json", type=str, default=str(_REPO_ROOT / "data/leetcode/eval_task_ids.json"))
    parser.add_argument("--buggy_tsvs", type=str, nargs="+", default=[
        str(_REPO_ROOT / "data/leetcode/test/q1_0_25.tsv"),
        str(_REPO_ROOT / "data/leetcode/test/q2_25_50.tsv"),
        str(_REPO_ROOT / "data/leetcode/test/q3_50_75.tsv"),
        str(_REPO_ROOT / "data/leetcode/test/q4_75_99.tsv"),
    ])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tester_py", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="newfacade/LeetCodeDataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_new_tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_fewshot", type=int, default=3)
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--tqdm_mininterval", type=float, default=5.0)
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--buggy_subset_split_mode", choices=["dev", "test", "all"], default="all",
                        help="Which deterministic half of each buggy TSV to evaluate. Use test for the second half.")
    parser.add_argument("--subset_split_seed", type=int, default=DEFAULT_SEED,
                        help="Seed for deterministic per-TSV half split. Default 1012 matches the reference script.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("testgen_eval")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_jsonl = output_dir / "per_problem_detailed.jsonl"
    summary_tsv = output_dir / "summary.tsv"

    set_seed(args.seed)
    tester = import_tester(args.tester_py)

    task_ids = load_task_ids(args.task_ids_json)
    if args.limit_tasks is not None:
        task_ids = task_ids[: args.limit_tasks]

    completed_keys: Set[Tuple[str, str, str]] = set()
    summary_rows: List[Dict[str, Any]] = []
    if args.resume:
        completed_keys = load_completed_keys(detail_jsonl)
        summary_rows = load_existing_summary(summary_tsv)
        logger.info("Resume enabled: found %d completed records.", len(completed_keys))
    else:
        if detail_jsonl.exists():
            detail_jsonl.unlink()
        if summary_tsv.exists():
            summary_tsv.unlink()

    logger.info(
        "Loading buggy rows from %d TSV files using deterministic %s half with seed=%d ...",
        len(args.buggy_tsvs),
        args.buggy_subset_split_mode,
        args.subset_split_seed,
    )
    buggy_df = load_buggy_rows(args.buggy_tsvs, split_mode=args.buggy_subset_split_mode, seed=args.subset_split_seed)
    buggy_df["task_id"] = buggy_df["task_id"].astype(str)
    buggy_df["question_id"] = buggy_df["question_id"].astype(str)

    logger.info("Loading dataset rows from split '%s' for %d task_ids...", args.split, len(task_ids))
    ds_rows = load_dataset_rows(args.dataset_name, args.split, task_ids)
    ds_map = {(str(r.get("task_id", "")), str(r.get("question_id", ""))): r for r in ds_rows}

    if args.peft_model_path:
        logger.info("Loading base model %s with PEFT adapter %s ...", args.model_name, args.peft_model_path)
    else:
        logger.info("Loading model %s ...", args.model_name)
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        dtype=args.dtype,
        peft_model_path=args.peft_model_path,
        merge_peft=args.merge_peft,
    )

    filtered_buggy_df = buggy_df[buggy_df["task_id"].isin(task_ids)].copy()
    logger.info(
        "Will evaluate %d buggy TSV rows matching task_ids after deterministic %s-half filtering.",
        len(filtered_buggy_df),
        args.buggy_subset_split_mode,
    )

    newly_processed = 0
    total_rows = len(filtered_buggy_df)

    if args.resume:
        remaining_rows = int(sum((str(r.task_id), str(r.question_id), str(r.source_tsv)) not in completed_keys for _, r in filtered_buggy_df.iterrows()))
    else:
        remaining_rows = total_rows

    progress = tqdm(
        total=remaining_rows,
        desc="Evaluating",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
        disable=args.disable_tqdm,
    )
    start_time = time.time()

    for _, buggy_row in filtered_buggy_df.iterrows():
        task_id = str(buggy_row["task_id"])
        question_id = str(buggy_row["question_id"])
        source_tsv = str(buggy_row.get("source_tsv", ""))
        resume_key = (task_id, question_id, source_tsv)
        if args.resume and resume_key in completed_keys:
            continue

        ds_row = ds_map.get((task_id, question_id))
        if ds_row is None:
            record = {
                "task_id": task_id,
                "question_id": question_id,
                "source_tsv": source_tsv,
                "status": "missing_dataset_row",
            }
            with detail_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed_keys.add(resume_key)
            continue

        fewshots = choose_fewshot_examples(ds_rows, args.num_fewshot, task_id)
        try:
            raw_generation, generated_test_code = generate_tests(
                model=model,
                tokenizer=tokenizer,
                row=ds_row,
                fewshots=fewshots,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            generation_error = ""
        except Exception as e:
            raw_generation = ""
            generated_test_code = ""
            generation_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        original_extracted_test_code = extract_python_block(generated_test_code)
        salvage_info = salvage_clean_test_code(raw_generation, original_extracted_test_code)
        cleaned_salvaged_test_code = salvage_info["clean_test_code"]

        gold_code = normalize_whitespace(ds_row.get("completion", ""))
        buggy_code = normalize_whitespace(buggy_row.get("prediction", ""))
        entry_point = tester.resolve_entry_point(ds_row.get("entry_point", ""), gold_code)
        prompt_text = ds_row.get("problem_description", "") or ds_row.get("query", "") or ""

        actual_buggy_pass_rate = None
        pass_rate_raw = buggy_row.get("pass_rate", None)
        if pass_rate_raw is not None and str(pass_rate_raw).strip() != "":
            try:
                actual_buggy_pass_rate = float(pass_rate_raw)
            except Exception:
                actual_buggy_pass_rate = None

        correctness = check_assert_correctness_against_gold(
            tester=tester,
            prompt_text=prompt_text,
            gold_code=gold_code,
            test_code=cleaned_salvaged_test_code,
            entry_point=entry_point,
        )

        generated_is_wellformed = bool(correctness.get("wellformed", False) and correctness.get("num_asserts", 0) > 0)
        if not generated_is_wellformed and not generation_error:
            parse_error = salvage_info.get("parse_error", "") or correctness.get("parse_error", "")
            generation_error = parse_error or "No valid asserts could be salvaged from the model output."

        if generated_is_wellformed:
            gold_eval = tester.eval_all_asserts_with_output(prompt_text, gold_code, cleaned_salvaged_test_code, entry_point)
            buggy_eval = tester.eval_all_asserts_with_output(prompt_text, buggy_code, cleaned_salvaged_test_code, entry_point)
            gold_results = gold_eval.get("results", [])
            buggy_results = buggy_eval.get("results", [])
            gold_passed = sum(1 for r in gold_results if r.get("ok"))
            buggy_passed = sum(1 for r in buggy_results if r.get("ok"))
            gold_total = int(gold_eval.get("total", len(gold_results)))
            buggy_total = int(buggy_eval.get("total", len(buggy_results)))
            gold_pass_rate = 100.0 * gold_passed / max(1, gold_total)
            buggy_pass_rate = 100.0 * buggy_passed / max(1, buggy_total)
            gold_eval_summary = {
                "total": gold_total,
                "passed": gold_passed,
                "failed": gold_total - gold_passed,
                "pass_rate": gold_pass_rate,
                "results": gold_results,
                "error": gold_eval.get("error", ""),
            }
            buggy_eval_summary = {
                "total": buggy_total,
                "passed": buggy_passed,
                "failed": buggy_total - buggy_passed,
                "pass_rate": buggy_pass_rate,
                "results": buggy_results,
                "error": buggy_eval.get("error", ""),
            }
            coverage_timeout = int(getattr(tester, "TIME_LIMIT_SEC", 10))
            gold_cov = compute_coverage(gold_code, cleaned_salvaged_test_code, entry_point, coverage_timeout)
            buggy_cov = compute_coverage(buggy_code, cleaned_salvaged_test_code, entry_point, coverage_timeout)
        else:
            err = generation_error or correctness.get("parse_error", "") or salvage_info.get("parse_error", "")
            gold_eval_summary = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "results": [], "error": err}
            buggy_eval_summary = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "results": [], "error": err}
            gold_cov = {"ok": False, "coverage_pct": 0.0, "executed_candidate_lines": [], "total_candidate_nonempty_lines": 0, "error": err}
            buggy_cov = {"ok": False, "coverage_pct": 0.0, "executed_candidate_lines": [], "total_candidate_nonempty_lines": 0, "error": err}

        predicted_buggy_pass_rate = buggy_eval_summary["pass_rate"]
        pass_rate_error = None if actual_buggy_pass_rate is None else predicted_buggy_pass_rate - actual_buggy_pass_rate

        record = {
            "task_id": task_id,
            "question_id": question_id,
            "source_tsv": source_tsv,
            "difficulty": buggy_row.get("difficulty", ds_row.get("difficulty", "")),
            "model_name": args.model_name,
            "peft_model_path": args.peft_model_path,
            "merge_peft": args.merge_peft,
            "entry_point": entry_point,
            "generation_error": generation_error,
            "generated_wellformed": generated_is_wellformed,
            "raw_generation": raw_generation,
            "generated_test_code_raw_extracted": original_extracted_test_code,
            "generated_test_code_clean": cleaned_salvaged_test_code,
            "salvaged_helper_lines": salvage_info["salvaged_helper_lines"],
            "salvaged_assert_blocks": salvage_info["salvaged_assert_blocks"],
            "num_salvaged_helper_lines": salvage_info["num_salvaged_helper_lines"],
            "num_salvaged_assert_blocks": salvage_info["num_salvaged_assert_blocks"],
            "salvage_dropped_lines": salvage_info["dropped_lines"],
            "salvage_parse_error": salvage_info["parse_error"],
            "salvage_mode": salvage_info["salvage_mode"],
            "gold_correctness": correctness,
            "gold_eval_on_generated_tests": gold_eval_summary,
            "buggy_eval_on_generated_tests": buggy_eval_summary,
            "gold_coverage_on_generated_tests": gold_cov,
            "buggy_coverage_on_generated_tests": buggy_cov,
            "actual_buggy_pass_rate_from_tsv": actual_buggy_pass_rate,
            "predicted_buggy_pass_rate_from_generated_tests": predicted_buggy_pass_rate,
            "buggy_pass_rate_error": pass_rate_error,
            "dataset_problem_description": ds_row.get("problem_description", ""),
            "dataset_starter_code": ds_row.get("starter_code", ""),
            "dataset_completion": gold_code,
            "buggy_prediction": buggy_code,
            "dataset_test": ds_row.get("test", ""),
            "fewshot_task_ids": [str(x.get("task_id", "")) for x in fewshots],
            "buggy_subset_split_mode": args.buggy_subset_split_mode,
            "subset_split_seed": args.subset_split_seed,
        }

        summary_row = {
            "task_id": task_id,
            "question_id": question_id,
            "generated_wellformed": generated_is_wellformed,
            "num_generated_asserts": correctness.get("num_asserts", 0),
            "correct_generated_asserts": correctness.get("correct_asserts", 0),
            "incorrect_generated_asserts": correctness.get("incorrect_asserts", 0),
            "gold_pass_rate_on_generated_tests": gold_eval_summary["pass_rate"],
            "buggy_pass_rate_on_generated_tests": buggy_eval_summary["pass_rate"],
            "gold_coverage_pct_on_generated_tests": gold_cov.get("coverage_pct", 0.0),
            "buggy_coverage_pct_on_generated_tests": buggy_cov.get("coverage_pct", 0.0),
            "actual_buggy_pass_rate_from_tsv": actual_buggy_pass_rate,
            "buggy_pass_rate_error": pass_rate_error,
            "source_tsv": source_tsv,
            "buggy_subset_split_mode": args.buggy_subset_split_mode,
            "subset_split_seed": args.subset_split_seed,
        }

        with detail_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed_keys.add(resume_key)
        summary_rows.append(summary_row)
        newly_processed += 1

        logger.info(
            "[%d] %s/%s | wellformed=%s | helpers=%d | asserts=%d | generated_buggy_pass=%.2f | actual_buggy_pass=%s | gold_cov=%.2f | buggy_cov=%.2f",
            len(summary_rows),
            task_id,
            question_id,
            generated_is_wellformed,
            salvage_info["num_salvaged_helper_lines"],
            salvage_info["num_salvaged_assert_blocks"],
            predicted_buggy_pass_rate,
            str(actual_buggy_pass_rate),
            gold_cov.get("coverage_pct", 0.0),
            buggy_cov.get("coverage_pct", 0.0),
        )
        progress.update(1)
        elapsed = max(1e-9, time.time() - start_time)
        rate = progress.n / elapsed
        remaining = max(0, progress.total - progress.n) if progress.total is not None else 0
        eta_seconds = remaining / rate if rate > 0 else float("inf")
        eta_min = eta_seconds / 60.0 if eta_seconds != float("inf") else float("inf")
        postfix = (
            f"task={task_id}/{question_id} "
            f"ok={generated_is_wellformed} "
            f"helpers={salvage_info['num_salvaged_helper_lines']} "
            f"asserts={salvage_info['num_salvaged_assert_blocks']} "
            f"gold_cov={gold_cov.get('coverage_pct', 0.0):.1f}% "
            f"buggy_pass={predicted_buggy_pass_rate:.1f}% "
            f"eta_min={eta_min:.1f}"
        )
        progress.set_postfix_str(postfix)

        if newly_processed % max(1, args.save_every) == 0:
            flush_outputs(output_dir, summary_rows, args, task_ids)
            logger.info("Flushed intermediate outputs after %d newly processed examples.", newly_processed)

    progress.close()
    flush_outputs(output_dir, summary_rows, args, task_ids)
    logger.info("Done.")


if __name__ == "__main__":
    main()