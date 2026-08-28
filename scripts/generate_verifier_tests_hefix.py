#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import logging
import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import multiprocessing as mp
import random
import re
import textwrap
import traceback
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from peft import PeftModel

SYSTEM_PROMPT = """You are a Python unit-test generation assistant for HumanEval-style programming problems.
Your job is to generate executable Python tests compatible with the archiki/UTGenDebug he_plus_fix split.

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
- Do not include text outside the Python code block.
- Do not include imports.
- Do not define helper functions.
- Do not copy is_floats(...) or assertion(...).
- Do not create inputs/results loops.
- Every assert must directly call candidate(...) or compare a candidate(...) result.
- Generate deterministic, executable asserts.
- Generate diverse tests: examples, edge cases, boundary cases, duplicates, negative/zero values, empty/singleton inputs when valid, and adversarial cases.
- The expected output must be correct with respect to the provided correct solution.
- If the correct solution returns None for an input, use None as the expected output.
- For floating-point outputs, use a direct tolerance assert such as:
  assert np.allclose(candidate(...), expected, rtol=1e-7, atol=1e-6)
  Numpy is already available as np in the evaluator.
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
import numpy as np
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
"""

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_PEFT_PATH = None
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "outputs/verifier_tests/hefix")
DEFAULT_SEED = 1012
MAX_FEWSHOT_TEST_CHARS = 6000


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
        f"Problem prompt:\n{normalize_whitespace(example['prompt'])}\n\n"
        f"Correct solution:\n{normalize_whitespace(example['canonical_solution'])}\n\n"
        f"Entry point:\n{example['entry_point']}\n\n"
        "Generate Python tests in the he_plus_fix dataset format."
    )


def truncate_for_prompt(text: str, max_chars: int = MAX_FEWSHOT_TEST_CHARS) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n    # ... truncated few-shot test suite for prompt length ..."


def make_fewshot_assistant(example: Dict[str, Any]) -> str:
    test_code = truncate_for_prompt(example["test"])
    return f"```python\n{test_code}\n```"


def build_prompt(tokenizer, row: Dict[str, Any], fewshot_examples: List[Dict[str, Any]]):
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": make_fewshot_user(ex)})
        messages.append({"role": "assistant", "content": make_fewshot_assistant(ex)})
    messages.append({
        "role": "user",
        "content": (
            f"Problem prompt:\n{normalize_whitespace(row['prompt'])}\n\n"
            f"Function signature:\n{normalize_whitespace(row.get('signature', ''))}\n\n"
            f"Correct solution:\n{normalize_whitespace(row['canonical_solution'])}\n\n"
            f"Entry point:\n{row['entry_point']}\n\n"
            "Generate Python tests in the he_plus_fix dataset format now. Generate as many deterministic tests as needed to aim for high coverage of the correct solution."
        ),
    })
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hefix_rows(dataset_name: str, split_name: str, limit_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name)
    if split_name not in ds:
        raise ValueError(f"Requested split '{split_name}' not found. Available splits: {list(ds.keys())}")
    rows = [dict(r) for r in ds[split_name]]
    if limit_tasks is not None:
        rows = rows[:limit_tasks]
    return rows


def choose_fewshot_examples(rows: List[Dict[str, Any]], k: int, exclude_task_id: str) -> List[Dict[str, Any]]:
    examples = [
        r for r in rows
        if str(r.get("task_id")) != str(exclude_task_id) and r.get("test") and r.get("canonical_solution")
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

def load_model_and_tokenizer(args):
    tokenizer_source = args.peft_model_path if args.peft_model_path else args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = resolve_torch_dtype(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=model_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if args.peft_model_path:
        model = PeftModel.from_pretrained(base_model, args.peft_model_path, is_trainable=False)
        if args.merge_peft:
            model = model.merge_and_unload()
    else:
        model = base_model
    model.eval()
    return model, tokenizer



def generate_tests(model, tokenizer, row: Dict[str, Any], fewshots: List[Dict[str, Any]], max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> Tuple[str, str]:
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
    return gen_text, extract_python_block(gen_text)


def _find_check_function(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            return node
    return None


def _node_mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _assert_mentions_candidate(assert_src: str) -> bool:
    try:
        tree = ast.parse(assert_src)
    except Exception:
        return False
    return any(isinstance(n, ast.Assert) and _node_mentions_name(n, "candidate") for n in ast.walk(tree))


def parse_asserts_from_check(test_code: str) -> List[ast.Assert]:
    """
    Extract asserts only from def check(candidate), not from helper functions.

    This is critical for HE+Fix because reference tests often contain helper
    functions such as assertion(out, exp, atol). If we count helper asserts,
    generated correctness metrics become invalid.
    """
    tree = ast.parse(normalize_whitespace(test_code))
    check_fn = _find_check_function(tree)
    if check_fn is None:
        return []
    out: List[ast.Assert] = []
    for stmt in check_fn.body:
        if isinstance(stmt, ast.Assert):
            out.append(stmt)
    out.sort(key=lambda x: (getattr(x, "lineno", 0), getattr(x, "col_offset", 0)))
    return out


def render_assert_source(assert_node: ast.Assert, source: str) -> str:
    seg = ast.get_source_segment(source, assert_node)
    return seg.strip() if seg else ""


def _parse_single_stmt(stmt_src: str) -> Optional[ast.stmt]:
    try:
        tree = ast.parse(stmt_src)
        if len(tree.body) != 1:
            return None
        return tree.body[0]
    except Exception:
        return None


def _is_safe_expr(expr_src: str) -> bool:
    try:
        tree = ast.parse(expr_src, mode="eval")
    except Exception:
        return False
    allowed_call_names = {"sorted", "tuple", "list", "set", "dict", "len", "sum", "min", "max", "abs", "range", "round"}
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
                ast.Expression, ast.Constant, ast.Name, ast.Load, ast.List, ast.Tuple, ast.Set, ast.Dict,
                ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.Invert,
                ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
                ast.BoolOp, ast.And, ast.Or,
                ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
                ast.Subscript, ast.Slice, ast.Call,
            )
            if not isinstance(node, allowed):
                self.ok = False
                return
            super().generic_visit(node)
    v = Visitor(); v.visit(tree)
    return bool(v.ok)


def _is_safe_helper_stmt(stmt_src: str) -> bool:
    stripped = stmt_src.strip()
    if not stripped:
        return False
    low = stripped.lower()
    if any(low.startswith(prefix) for prefix in DISALLOWED_HELPER_PREFIXES):
        return False
    if CHECK_HEADER_RE.match(stripped) or stripped.startswith("#"):
        return False
    stmt = _parse_single_stmt(stripped)
    if stmt is None or not isinstance(stmt, ast.Assign):
        return False
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
        return False
    m = SAFE_ASSIGN_RE.match(stripped)
    return bool(m and _is_safe_expr(m.group(2)))


def _count_balance(text: str) -> int:
    score = 0
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {")", "]", "}"}
    in_str = False; quote = ""; esc = False
    for ch in text:
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == quote: in_str = False
            continue
        if ch in {"'", '"'}:
            in_str = True; quote = ch; continue
        if ch in pairs: score += 1
        elif ch in closing: score -= 1
    return score


def _extract_candidate_asserts_from_parsed_check(src: str) -> Tuple[List[str], List[str], str]:
    """
    AST-first extraction: keep only top-level asserts inside def check(candidate)
    that mention candidate. Drop helper functions and helper-body asserts.
    """
    dropped: List[str] = []
    try:
        tree = ast.parse(normalize_whitespace(src))
    except Exception as e:
        return [], [], f"{type(e).__name__}: {e}"

    check_fn = _find_check_function(tree)
    if check_fn is None:
        return [], [], "no check function"

    asserts: List[str] = []
    for stmt in check_fn.body:
        seg = ast.get_source_segment(src, stmt) or ""
        seg = seg.strip()
        if isinstance(stmt, ast.Assert):
            if _node_mentions_name(stmt, "candidate"):
                asserts.append(seg)
            else:
                dropped.append(seg or ast.unparse(stmt))
        elif seg:
            dropped.append(seg)

    return asserts, dropped, ""


def _extract_check_block_text(src: str) -> str:
    """
    Best-effort line fallback for syntactically truncated generations.
    It returns only the indented body of def check(candidate).
    """
    lines = strip_markdown_fences(src).splitlines()
    start = None
    header_indent = 0
    for i, line in enumerate(lines):
        if CHECK_HEADER_RE.match(line.strip()):
            start = i + 1
            header_indent = len(line) - len(line.lstrip(" "))
            break
    if start is None:
        return strip_markdown_fences(src)

    body: List[str] = []
    for line in lines[start:]:
        if line.strip() == "":
            body.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= header_indent and not line.lstrip().startswith("#"):
            break
        body.append(line)
    return "\n".join(body)


def salvage_structured_blocks(text: str) -> Dict[str, Any]:
    src = strip_markdown_fences(text)

    # First try proper AST extraction from def check(candidate).
    parsed_asserts, parsed_dropped, parse_err = _extract_candidate_asserts_from_parsed_check(src)
    if parsed_asserts:
        seen: Set[str] = set()
        asserts: List[str] = []
        for a in parsed_asserts:
            if a not in seen:
                seen.add(a)
                asserts.append(a)
        clean_lines = ["def check(candidate):"]
        for a in asserts:
            clean_lines.extend("    " + ln for ln in a.splitlines())
        clean_test_code = "\n".join(clean_lines)
        clean_parse_error = ""
        try:
            ast.parse(clean_test_code)
        except Exception as e:
            clean_parse_error = f"{type(e).__name__}: {e}"
        return {
            "clean_test_code": clean_test_code,
            "helper_lines": [],
            "assert_blocks": asserts,
            "num_helper_lines": 0,
            "num_assert_blocks": len(asserts),
            "dropped_lines": parsed_dropped,
            "parse_error": clean_parse_error,
        }

    # Fallback for truncated/incomplete model output: scan only inside check body.
    check_body_text = _extract_check_block_text(src)
    lines = check_body_text.splitlines()
    asserts, dropped = [], []
    seen_asserts: Set[str] = set()
    i = 0
    while i < len(lines):
        stripped = lines[i].rstrip("\n").strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            i += 1
            continue

        if ASSERT_START_RE.match(stripped):
            block_lines = [stripped]
            j = i + 1
            while True:
                joined = "\n".join(block_lines)
                stmt = _parse_single_stmt(joined)

                if isinstance(stmt, ast.Assert):
                    if _assert_mentions_candidate(joined):
                        if joined not in seen_asserts:
                            seen_asserts.add(joined)
                            asserts.append(joined)
                    else:
                        dropped.append(joined)
                    i = j
                    break

                if j >= len(lines):
                    dropped.append(joined)
                    i = j
                    break

                nxt = lines[j].rstrip("\n").strip()
                if ASSERT_START_RE.match(nxt) and _count_balance(joined) <= 0:
                    dropped.append(joined)
                    i = j
                    break
                if nxt.startswith("```"):
                    dropped.append(joined)
                    i = j + 1
                    break

                block_lines.append(nxt)
                j += 1
            continue

        dropped.append(stripped)
        i += 1

    clean_lines = ["def check(candidate):"]
    if not asserts:
        clean_lines.append("    pass")
    else:
        for a in asserts:
            clean_lines.extend("    " + ln for ln in a.splitlines())

    clean_test_code = "\n".join(clean_lines)
    clean_parse_error = ""
    try:
        ast.parse(clean_test_code)
    except Exception as e:
        clean_parse_error = f"{type(e).__name__}: {e}"

    return {
        "clean_test_code": clean_test_code,
        "helper_lines": [],
        "assert_blocks": asserts,
        "num_helper_lines": 0,
        "num_assert_blocks": len(asserts),
        "dropped_lines": parsed_dropped + dropped,
        "parse_error": clean_parse_error or parse_err,
    }


def salvage_clean_test_code(raw_generation: str, extracted_test_code: str) -> Dict[str, Any]:
    first = salvage_structured_blocks(extracted_test_code)
    second = salvage_structured_blocks(raw_generation)

    asserts, seen_a = [], set()
    for item in first["assert_blocks"] + second["assert_blocks"]:
        if item not in seen_a:
            seen_a.add(item)
            asserts.append(item)

    clean_lines = ["def check(candidate):"]
    if not asserts:
        clean_lines.append("    pass")
    else:
        for a in asserts:
            clean_lines.extend("    " + ln for ln in a.splitlines())

    clean_test_code = "\n".join(clean_lines)
    parse_error = ""
    try:
        ast.parse(clean_test_code)
    except Exception as e:
        parse_error = f"{type(e).__name__}: {e}"

    return {
        "clean_test_code": clean_test_code,
        "salvaged_helper_lines": [],
        "salvaged_assert_blocks": asserts,
        "num_salvaged_helper_lines": 0,
        "num_salvaged_assert_blocks": len(asserts),
        "dropped_lines": first["dropped_lines"] + second["dropped_lines"],
        "parse_error": parse_error,
        "salvage_mode": "check_ast_or_check_body_only",
    }


class _AssertCounter(ast.NodeVisitor):
    def __init__(self): self.count = 0
    def visit_Assert(self, node): self.count += 1; self.generic_visit(node)


def count_asserts(test_code: str) -> int:
    try: tree = ast.parse(normalize_whitespace(test_code))
    except Exception: return 0
    c = _AssertCounter(); c.visit(tree); return c.count


class _CheckToCollector(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name != "check":
            return node

        node.name = "__collect_asserts"
        new_body = []

        # Keep only top-level assert statements. Do not preserve helper
        # assignments or other statements; they can crash thunk collection
        # before any candidate assert is materialized.
        for stmt in node.body:
            if isinstance(stmt, ast.Assert):
                lam = ast.Lambda(
                    args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                    body=stmt.test,
                )
                call = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="__ASSERTS__", ctx=ast.Load()),
                        attr="append",
                        ctx=ast.Load(),
                    ),
                    args=[lam],
                    keywords=[],
                )
                new_body.append(ast.copy_location(ast.Expr(value=call), stmt))

        if not new_body:
            new_body = [ast.Pass()]

        node.body = new_body
        return node


def thunkify_tests(test_code: str) -> str:
    tree = ast.parse(normalize_whitespace(test_code))
    tree = _CheckToCollector().visit(tree)
    ast.fix_missing_locations(tree)
    prelude = ast.parse("__ASSERTS__ = globals().get('__ASSERTS__', [])\n")
    module = ast.Module(body=prelude.body + tree.body, type_ignores=[])
    return ast.unparse(module)


def resolve_entry_point(ep: str | None, completion_code: str) -> str:
    if isinstance(ep, str) and ep.strip():
        return ep.strip()
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", completion_code, re.M)
    return m.group(1) if m else "solve"


def make_full_candidate_code(prompt_text: str, code_or_body: str, entry_point: str) -> str:
    """
    HE+Fix `canonical_solution` is a full function, but HE+Fix `code` is usually
    only the buggy function body. Wrap body-only code with the original prompt so
    the requested entry point actually exists during evaluation.
    """
    code_or_body = normalize_whitespace(code_or_body)
    prompt_text = normalize_whitespace(prompt_text)

    if not code_or_body:
        return prompt_text + "\n    pass\n"

    ep = (entry_point or "").strip()
    if re.search(rf"^\s*def\s+{re.escape(ep)}\s*\(", code_or_body, re.M):
        return code_or_body

    if re.search(r"^\s*def\s+[A-Za-z_]\w*\s*\(", code_or_body, re.M):
        return code_or_body

    if not prompt_text:
        return code_or_body

    # The prompt contains imports + the function signature/docstring.
    # Append the body indented by one function level.
    body = textwrap.indent(code_or_body.rstrip() + "\n", "    ")
    return prompt_text.rstrip() + "\n" + body


def build_test_code_from_correct_asserts(correctness: Dict[str, Any]) -> str:
    kept: List[str] = []
    seen: Set[str] = set()
    for d in correctness.get("assert_details", []):
        src = str(d.get("assert_src", "")).strip()
        if d.get("is_correct_wrt_gold") and src and "candidate" in src and src not in seen:
            seen.add(src)
            kept.append(src)

    lines = ["def check(candidate):"]
    if not kept:
        lines.append("    pass")
    else:
        for a in kept:
            lines.extend("    " + ln for ln in a.splitlines())
    return "\n".join(lines)


def eval_all_asserts_with_output(prompt_text: str, completion: str, test_code: str, entry_point: str, timeout: int = 10) -> Dict[str, Any]:
    expected_total = count_asserts(test_code)
    if expected_total <= 0:
        return {"total": 0, "results": [], "error": "no_asserts"}
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=1)
    def _proc(q_):
        try:
            comp = normalize_whitespace(completion); tests = normalize_whitespace(test_code); ep = resolve_entry_point(entry_point, comp)
            thunk_src = thunkify_tests(tests)
            full = f"""{PROMPT_IMPORTS}
__ASSERTS__ = []
__CANDIDATE_LOAD_ERR__ = ""
def __stub__(*args, **kwargs):
    raise NotImplementedError("entry point is missing or invalid")
def __get_entry_point():
    if __CANDIDATE_LOAD_ERR__:
        return __stub__
    try:
        return {ep}
    except Exception:
        return __stub__
try:
    exec({json.dumps(comp)}, globals(), globals())
except Exception as e:
    __CANDIDATE_LOAD_ERR__ = f"candidate_syntax_or_exec: {{type(e).__name__}}: {{e}}"
{thunk_src}
try:
    __collect_asserts(__get_entry_point())
except Exception:
    pass
"""
            g: Dict[str, Any] = {}
            exec(compile(full, "<eval_generated_asserts>", "exec"), g, g)
            out = {"total": expected_total, "results": [], "candidate_load_error": g.get("__CANDIDATE_LOAD_ERR__", "")}
            for idx, th in enumerate(list(g.get("__ASSERTS__", []))):
                buf = io.StringIO(); ok = False; err_type = ""; err_msg = ""
                with contextlib.redirect_stdout(buf):
                    try: ok = bool(th())
                    except Exception as e:
                        ok = False; err_type = type(e).__name__; err_msg = str(e)
                out["results"].append({"index": idx, "ok": bool(ok), "stdout": buf.getvalue(), "error_type": err_type, "error_msg": err_msg[:2000]})
            for idx in range(len(out["results"]), expected_total):
                out["results"].append({"index": idx, "ok": False, "stdout": "", "error_type": "missing_thunk", "error_msg": "collector did not create this assert thunk"})
            q_.put(out)
        except Exception as e:
            q_.put({"total": expected_total, "results": [], "error": f"{type(e).__name__}: {e}"})
    p = ctx.Process(target=_proc, args=(q,)); p.start(); p.join(timeout)
    if p.is_alive():
        try: p.terminate()
        finally: p.join(1)
        return {"total": expected_total, "results": [], "error": f"timeout >{timeout}s"}
    try: return q.get_nowait()
    except Exception: return {"total": expected_total, "results": [], "error": "no result"}


def eval_single_assert_with_output(prompt_text: str, completion: str, test_code: str, entry_point: str, assert_index: int, timeout: int = 10) -> Dict[str, Any]:
    allres = eval_all_asserts_with_output(prompt_text, completion, test_code, entry_point, timeout)
    results = allres.get("results", [])
    if assert_index < 0 or assert_index >= len(results):
        return {"ok": False, "stdout": "", "error_type": "index", "error_msg": str(allres.get("error", "index out of range"))}
    r = results[assert_index]
    return {"ok": bool(r.get("ok")), "stdout": r.get("stdout", ""), "error_type": r.get("error_type", ""), "error_msg": r.get("error_msg", "")}


def check_assert_correctness_against_gold(prompt_text: str, gold_code: str, test_code: str, entry_point: str) -> Dict[str, Any]:
    src = normalize_whitespace(test_code)
    try:
        asserts = parse_asserts_from_check(src)
    except Exception as e:
        return {"wellformed": False, "parse_error": f"{type(e).__name__}: {e}", "num_asserts": 0, "correct_asserts": 0, "incorrect_asserts": 0, "assert_details": []}
    details, correct, incorrect = [], 0, 0
    for idx, a in enumerate(asserts):
        assert_src = render_assert_source(a, src)
        res = eval_single_assert_with_output(prompt_text, gold_code, src, entry_point, idx)
        is_correct = bool(res.get("ok", False))
        correct += int(is_correct); incorrect += int(not is_correct)
        details.append({"index": idx, "assert_src": assert_src, "is_correct_wrt_gold": is_correct,
                        "error_type": res.get("error_type", ""), "error_msg": res.get("error_msg", ""), "stdout": res.get("stdout", "")})
    return {"wellformed": True, "parse_error": "", "num_asserts": len(asserts), "correct_asserts": correct, "incorrect_asserts": incorrect, "assert_details": details}


def extract_inputs_results_from_hefix_test(test_code: str) -> Tuple[Optional[List[Any]], Optional[List[Any]], str]:
    try:
        tree = ast.parse(normalize_whitespace(test_code))
        check_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
        if check_fn is None:
            return None, None, "no check function"
        vals: Dict[str, Any] = {}
        for node in ast.walk(check_fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in {"inputs", "results"}:
                        vals[target.id] = ast.literal_eval(node.value)
        if "inputs" in vals and "results" in vals and len(vals["inputs"]) == len(vals["results"]):
            return vals["inputs"], vals["results"], ""
        return None, None, "inputs/results not found or length mismatch"
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def eval_hefix_reference_tests(prompt_text: str, completion: str, test_code: str, entry_point: str, timeout: int = 10) -> Dict[str, Any]:
    inputs, results, parse_error = extract_inputs_results_from_hefix_test(test_code)
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=1)
    def _proc(q_):
        try:
            comp = normalize_whitespace(completion); tests = normalize_whitespace(test_code); ep = resolve_entry_point(entry_point, comp)
            full = f"""{PROMPT_IMPORTS}
try:
    exec({json.dumps(comp)}, globals(), globals())
except Exception as e:
    __LOAD_ERR__ = f"candidate_syntax_or_exec: {{type(e).__name__}}: {{e}}"
else:
    __LOAD_ERR__ = ""
{tests}
try:
    __CANDIDATE__ = {ep}
except Exception as e:
    __LOAD_ERR__ = __LOAD_ERR__ or f"entry_point: {{type(e).__name__}}: {{e}}"
    def __CANDIDATE__(*args, **kwargs):
        raise NotImplementedError("entry point missing")
"""
            g: Dict[str, Any] = {}
            exec(compile(full, "<eval_hefix_ref>", "exec"), g, g)
            candidate = g["__CANDIDATE__"]
            if inputs is not None and results is not None:
                out = {"total": len(inputs), "results": [], "mode": "parsed_inputs_results", "parse_error": parse_error, "candidate_load_error": g.get("__LOAD_ERR__", "")}
                assertion_fn = g.get("assertion")
                for idx, (inp, exp) in enumerate(zip(inputs, results)):
                    buf = io.StringIO(); ok = False; err_type = ""; err_msg = ""
                    with contextlib.redirect_stdout(buf):
                        try:
                            args = inp if isinstance(inp, (list, tuple)) else [inp]
                            got = candidate(*args)
                            if callable(assertion_fn):
                                assertion_fn(got, exp, 0)
                            else:
                                if isinstance(exp, float):
                                    assert np.allclose(got, exp, rtol=1e-7, atol=1e-6)
                                else:
                                    assert got == exp
                            ok = True
                        except Exception as e:
                            ok = False; err_type = type(e).__name__; err_msg = str(e)
                    out["results"].append({"index": idx, "ok": bool(ok), "stdout": buf.getvalue(), "error_type": err_type, "error_msg": err_msg[:2000]})
                q_.put(out)
            else:
                buf = io.StringIO(); ok = False; err_type = ""; err_msg = ""
                with contextlib.redirect_stdout(buf):
                    try:
                        g["check"](candidate); ok = True
                    except Exception as e:
                        ok = False; err_type = type(e).__name__; err_msg = str(e)
                q_.put({"total": 1, "results": [{"index": 0, "ok": ok, "stdout": buf.getvalue(), "error_type": err_type, "error_msg": err_msg[:2000]}], "mode": "whole_check_fallback", "parse_error": parse_error, "candidate_load_error": g.get("__LOAD_ERR__", "")})
        except Exception as e:
            q_.put({"total": 0, "results": [], "error": f"{type(e).__name__}: {e}", "mode": "exec_error", "parse_error": parse_error})
    p = ctx.Process(target=_proc, args=(q,)); p.start(); p.join(timeout)
    if p.is_alive():
        try: p.terminate()
        finally: p.join(1)
        total = len(inputs) if inputs is not None else 1
        return {"total": total, "results": [], "error": f"timeout >{timeout}s", "mode": "timeout", "parse_error": parse_error}
    try: return q.get_nowait()
    except Exception: return {"total": 0, "results": [], "error": "no result", "mode": "no_result", "parse_error": parse_error}


def compute_pass_summary(eval_res: Dict[str, Any]) -> Dict[str, Any]:
    results = eval_res.get("results", [])
    total = int(eval_res.get("total", len(results)))
    passed = sum(1 for r in results if r.get("ok"))
    failed = max(0, total - passed)
    pass_rate = 100.0 * passed / max(1, total)
    return {"total": total, "passed": passed, "failed": failed, "pass_rate": pass_rate, "results": results, "error": eval_res.get("error", "")}


def _coverage_worker(q, candidate_code: str, test_code: str, entry_point: str, timeout: int):
    try:
        import sys
        candidate_code = normalize_whitespace(candidate_code)
        test_code = normalize_whitespace(test_code)
        ep = resolve_entry_point(entry_point, candidate_code)

        full = f"""{PROMPT_IMPORTS}
{candidate_code}
candidate = {ep}
{test_code}
check(candidate)
"""
        filename = "<hefix_coverage>"
        executed_lines: Set[int] = set()

        def _trace(frame, event, arg):
            if event == "line" and frame.f_code.co_filename == filename:
                executed_lines.add(frame.f_lineno)
            return _trace

        glb: Dict[str, Any] = {"__name__": "__main__"}
        ok = True
        err = ""
        try:
            code = compile(full, filename, "exec")
            with contextlib.redirect_stdout(io.StringIO()):
                sys.settrace(_trace)
                exec(code, glb, glb)
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {e}"
        finally:
            sys.settrace(None)

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

        q.put({
            "ok": ok,
            "coverage_pct": 100.0 * len(candidate_executed) / max(1, total_lines),
            "executed_candidate_lines": candidate_executed,
            "total_candidate_nonempty_lines": total_lines,
            "error": err,
        })
    except Exception as e:
        q.put({
            "ok": False,
            "coverage_pct": 0.0,
            "executed_candidate_lines": [],
            "total_candidate_nonempty_lines": 0,
            "error": f"{type(e).__name__}: {e}",
        })


def compute_coverage(candidate_code: str, test_code: str, entry_point: str, timeout: int) -> Dict[str, Any]:
    ctx = mp.get_context("fork"); q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_coverage_worker, args=(q, candidate_code, test_code, entry_point, timeout))
    p.start(); p.join(timeout)
    if p.is_alive():
        try: p.terminate()
        finally: p.join(1)
        return {"ok": False, "coverage_pct": 0.0, "executed_candidate_lines": [], "total_candidate_nonempty_lines": 0, "error": f"timeout >{timeout}s"}
    try: return q.get_nowait()
    except Exception: return {"ok": False, "coverage_pct": 0.0, "executed_candidate_lines": [], "total_candidate_nonempty_lines": 0, "error": "no result"}


def load_completed_task_ids(detail_jsonl: Path) -> Set[str]:
    done: Set[str] = set()
    if not detail_jsonl.exists(): return done
    with detail_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("task_id"): done.add(str(rec["task_id"]))
            except Exception:
                continue
    return done


def load_existing_summary(summary_tsv: Path) -> List[Dict[str, Any]]:
    if not summary_tsv.exists(): return []
    try: return pd.read_csv(summary_tsv, sep="\t").to_dict(orient="records")
    except Exception: return []


def flush_outputs(output_dir: Path, summary_rows: List[Dict[str, Any]], args: argparse.Namespace, total_dataset_rows: int) -> None:
    summary_df = pd.DataFrame(summary_rows)
    summary_tsv = output_dir / "summary.tsv"
    summary_df.to_csv(summary_tsv, sep="\t", index=False)
    bug_err = summary_df["buggy_pass_rate_error"].dropna() if not summary_df.empty and "buggy_pass_rate_error" in summary_df.columns else pd.Series(dtype=float)
    agg = {
        "model_name": args.model_name,
        "peft_model_path": args.peft_model_path,
        "merge_peft": args.merge_peft,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "num_dataset_rows": total_dataset_rows,
        "num_rows_evaluated": int(len(summary_df)),
        "num_wellformed_generations": int(summary_df["generated_wellformed"].sum()) if not summary_df.empty else 0,
        "wellformed_rate": float(summary_df["generated_wellformed"].mean()) if not summary_df.empty else 0.0,
        "avg_num_generated_asserts": float(summary_df["num_generated_asserts"].mean()) if not summary_df.empty else 0.0,
        "avg_correct_generated_asserts": float(summary_df["correct_generated_asserts"].mean()) if not summary_df.empty else 0.0,
        "avg_incorrect_generated_asserts": float(summary_df["incorrect_generated_asserts"].mean()) if not summary_df.empty else 0.0,
        "avg_gold_pass_rate_on_generated_tests": float(summary_df["gold_pass_rate_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_buggy_pass_rate_on_generated_tests": float(summary_df["buggy_pass_rate_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_gold_coverage_pct_on_generated_tests": float(summary_df["gold_coverage_pct_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_buggy_coverage_pct_on_generated_tests": float(summary_df["buggy_coverage_pct_on_generated_tests"].mean()) if not summary_df.empty else 0.0,
        "avg_actual_buggy_pass_rate_from_dataset_tests": float(summary_df["actual_buggy_pass_rate_from_dataset_tests"].dropna().mean()) if not summary_df.empty else 0.0,
        "mean_buggy_pass_rate_error": float(bug_err.mean()) if not bug_err.empty else 0.0,
        "mae_buggy_pass_rate_error": float(bug_err.abs().mean()) if not bug_err.empty else 0.0,
        "files": {"per_problem_detailed_jsonl": str(output_dir / "per_problem_detailed.jsonl"), "summary_tsv": str(summary_tsv)},
        "config": vars(args),
    }
    with (output_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate Qwen HE+Fix tests on archiki/UTGenDebug he_plus_fix.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--peft_model_path", type=str, default=DEFAULT_PEFT_PATH)
    parser.add_argument("--merge_peft", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset_name", type=str, default="archiki/UTGenDebug")
    parser.add_argument("--split", type=str, default="he_plus_fix")
    parser.add_argument("--max_new_tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num_fewshot", type=int, default=3)
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--eval_timeout", type=int, default=10)
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--tqdm_mininterval", type=float, default=5.0)
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--evaluate_raw_generated_tests", action="store_true", default=False,
                        help="Use all salvaged generated asserts for buggy evaluation. Default uses only asserts that pass on gold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("hefix_testgen_eval")
    out_root = Path(args.output_dir)
    output_dir = out_root
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_jsonl = output_dir / "per_problem_detailed.jsonl"
    summary_tsv = output_dir / "summary.tsv"

    set_seed(args.seed)
    rows = load_hefix_rows(args.dataset_name, args.split, args.limit_tasks)
    total_dataset_rows = len(rows)

    completed = set(); summary_rows: List[Dict[str, Any]] = []
    if args.resume:
        completed = load_completed_task_ids(detail_jsonl)
        summary_rows = load_existing_summary(summary_tsv)
        logger.info("Resume enabled: found %d completed task_ids.", len(completed))
    else:
        if detail_jsonl.exists(): detail_jsonl.unlink()
        if summary_tsv.exists(): summary_tsv.unlink()

    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(args)

    remaining = sum(str(r.get("task_id", "")) not in completed for r in rows)
    progress = tqdm(total=remaining, desc="Evaluating HE+Fix", dynamic_ncols=True, mininterval=args.tqdm_mininterval, disable=args.disable_tqdm)
    newly_processed = 0
    start_time = time.time()

    for row in rows:
        task_id = str(row.get("task_id", ""))
        if args.resume and task_id in completed:
            continue
        question_id = task_id
        fewshots = choose_fewshot_examples(rows, args.num_fewshot, task_id)
        try:
            raw_generation, generated_test_code = generate_tests(model, tokenizer, row, fewshots, args.max_new_tokens, args.do_sample, args.temperature, args.top_p)
            generation_error = ""
        except Exception as e:
            raw_generation = ""; generated_test_code = ""
            generation_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        original_extracted_test_code = extract_python_block(generated_test_code)
        salvage_info = salvage_clean_test_code(raw_generation, original_extracted_test_code)
        cleaned_test_code = salvage_info["clean_test_code"]

        prompt_text = row.get("prompt", "")
        gold_code = normalize_whitespace(row.get("canonical_solution", ""))
        entry_point = resolve_entry_point(row.get("entry_point", ""), gold_code)
        buggy_code_body = normalize_whitespace(row.get("code", ""))
        buggy_code = make_full_candidate_code(prompt_text, buggy_code_body, entry_point)
        dataset_test = row.get("test", "")

        correctness = check_assert_correctness_against_gold(prompt_text, gold_code, cleaned_test_code, entry_point)
        eval_test_code = cleaned_test_code if args.evaluate_raw_generated_tests else build_test_code_from_correct_asserts(correctness)
        num_eval_asserts = count_asserts(eval_test_code)
        generated_is_wellformed = bool(correctness.get("wellformed", False) and correctness.get("num_asserts", 0) > 0)
        generated_is_usable = bool(generated_is_wellformed and num_eval_asserts > 0)
        if not generated_is_wellformed and not generation_error:
            generation_error = salvage_info.get("parse_error", "") or correctness.get("parse_error", "") or "No valid asserts could be salvaged from the model output."
        if generated_is_wellformed and not generated_is_usable and not generation_error:
            generation_error = "No generated asserts passed on the gold solution; generated tests are not usable for buggy pass-rate estimation."

        original_gold_eval = compute_pass_summary(eval_hefix_reference_tests(prompt_text, gold_code, dataset_test, entry_point, args.eval_timeout))
        original_buggy_eval = compute_pass_summary(eval_hefix_reference_tests(prompt_text, buggy_code, dataset_test, entry_point, args.eval_timeout))
        actual_buggy_pass_rate = original_buggy_eval["pass_rate"]

        if generated_is_usable:
            gold_eval_summary = compute_pass_summary(eval_all_asserts_with_output(prompt_text, gold_code, eval_test_code, entry_point, args.eval_timeout))
            buggy_eval_summary = compute_pass_summary(eval_all_asserts_with_output(prompt_text, buggy_code, eval_test_code, entry_point, args.eval_timeout))
            gold_cov = compute_coverage(gold_code, eval_test_code, entry_point, args.eval_timeout)
            buggy_cov = compute_coverage(buggy_code, eval_test_code, entry_point, args.eval_timeout)
        else:
            err = generation_error
            gold_eval_summary = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "results": [], "error": err}
            buggy_eval_summary = {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "results": [], "error": err}
            gold_cov = {"ok": False, "coverage_pct": 0.0, "executed_candidate_lines": [], "total_candidate_nonempty_lines": 0, "error": err}
            buggy_cov = {"ok": False, "coverage_pct": 0.0, "executed_candidate_lines": [], "total_candidate_nonempty_lines": 0, "error": err}

        predicted_buggy_pass_rate = buggy_eval_summary["pass_rate"]
        pass_rate_error = predicted_buggy_pass_rate - actual_buggy_pass_rate

        record = {
            "task_id": task_id,
            "question_id": question_id,
            "model_name": args.model_name,
            "peft_model_path": args.peft_model_path,
            "merge_peft": args.merge_peft,
            "entry_point": entry_point,
            "generation_error": generation_error,
            "generated_wellformed": generated_is_wellformed,
            "raw_generation": raw_generation,
            "generated_test_code_raw_extracted": original_extracted_test_code,
            "generated_test_code_clean": cleaned_test_code,
            "generated_test_code_eval": eval_test_code,
            "num_eval_asserts": num_eval_asserts,
            "evaluate_raw_generated_tests": args.evaluate_raw_generated_tests,
            "generated_usable": generated_is_usable,
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
            "original_gold_eval_on_dataset_tests": original_gold_eval,
            "original_buggy_eval_on_dataset_tests": original_buggy_eval,
            "actual_buggy_pass_rate_from_dataset_tests": actual_buggy_pass_rate,
            "predicted_buggy_pass_rate_from_generated_tests": predicted_buggy_pass_rate,
            "buggy_pass_rate_error": pass_rate_error,
            "dataset_prompt": prompt_text,
            "dataset_signature": row.get("signature", ""),
            "dataset_canonical_solution": gold_code,
            "buggy_code_body": buggy_code_body,
            "buggy_code": buggy_code,
            "dataset_test": dataset_test,
            "fewshot_task_ids": [str(x.get("task_id", "")) for x in fewshots],
        }
        summary_row = {
            "task_id": task_id,
            "question_id": question_id,
            "generated_wellformed": generated_is_wellformed,
            "num_generated_asserts": correctness.get("num_asserts", 0),
            "num_eval_asserts": num_eval_asserts,
            "generated_usable": generated_is_usable,
            "correct_generated_asserts": correctness.get("correct_asserts", 0),
            "incorrect_generated_asserts": correctness.get("incorrect_asserts", 0),
            "gold_pass_rate_on_generated_tests": gold_eval_summary["pass_rate"],
            "buggy_pass_rate_on_generated_tests": buggy_eval_summary["pass_rate"],
            "gold_coverage_pct_on_generated_tests": gold_cov.get("coverage_pct", 0.0),
            "buggy_coverage_pct_on_generated_tests": buggy_cov.get("coverage_pct", 0.0),
            "original_gold_pass_rate_on_dataset_tests": original_gold_eval["pass_rate"],
            "actual_buggy_pass_rate_from_dataset_tests": actual_buggy_pass_rate,
            "buggy_pass_rate_error": pass_rate_error,
        }
        with detail_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        completed.add(task_id); summary_rows.append(summary_row); newly_processed += 1
        logger.info("[%d] %s | wellformed=%s | asserts=%d | gold_gen_pass=%.2f | buggy_gen_pass=%.2f | actual_buggy=%.2f | err=%.2f",
                    len(summary_rows), task_id, generated_is_wellformed, correctness.get("num_asserts", 0),
                    gold_eval_summary["pass_rate"], predicted_buggy_pass_rate, actual_buggy_pass_rate, pass_rate_error)
        progress.update(1)
        elapsed = max(1e-9, time.time() - start_time); rate = progress.n / elapsed
        rem = max(0, progress.total - progress.n) if progress.total is not None else 0
        eta_min = rem / rate / 60.0 if rate > 0 else float("inf")
        progress.set_postfix_str(f"task={task_id} ok={generated_is_wellformed} asserts={correctness.get('num_asserts', 0)} gen_buggy={predicted_buggy_pass_rate:.1f}% actual={actual_buggy_pass_rate:.1f}% eta_min={eta_min:.1f}")
        if newly_processed % max(1, args.save_every) == 0:
            flush_outputs(output_dir, summary_rows, args, total_dataset_rows)
            logger.info("Flushed intermediate outputs after %d newly processed examples.", newly_processed)
    progress.close()
    flush_outputs(output_dir, summary_rows, args, total_dataset_rows)
    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
