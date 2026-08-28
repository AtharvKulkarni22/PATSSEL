#!/usr/bin/env python3
# eval_thunked.py
#
# Training-focused evaluator:
# - Thunkified tests (each assert becomes an independent thunk; no early abort)
# - Deterministic totals from AST assert count
# - Candidate code is loaded safely; syntax/runtime errors don’t crash harness
# - Missing/invalid entry point is stubbed
# - Subprocess with hard timeout per evaluation
#
# Exposed API:
#   - PROMPT_IMPORTS, _dedent, resolve_entry_point
#   - count_asserts, thunkify_tests
#   - eval_with_timeout(prompt, completion, test_code, entry_point, timeout=None) -> dict
#   - evaluate_code(prompt, completion, test_code, entry_point) -> dict
#
# Return schema: dict(passed, failed, total, ok, error_type, error_msg)

import os
import re
import ast
import json
import textwrap
import multiprocessing as mp
from typing import Any, Dict

# ======== Config
TIME_LIMIT_SEC = int(os.getenv("EVAL_TIMEOUT", "10"))

# ======== Helpers (from original test_exec.py)
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

def _dedent(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return textwrap.dedent(s.replace("\r\n", "\n").replace("\r", "\n")).lstrip("\ufeff")

def resolve_entry_point(ep: str | None, completion_code: str) -> str:
    if isinstance(ep, str) and ep.strip():
        return ep.strip()
    m = re.search(r"class\s+Solution\s*:\s*.*?\n\s*def\s+([A-Za-z_]\w*)\s*\(", completion_code, re.S)
    if m:
        return f"Solution().{m.group(1)}"
    m = re.search(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", completion_code, re.M)
    return m.group(1) if m else "solve"

# ======== Deterministic assert counting (AST)
class _AssertCounter(ast.NodeVisitor):
    def __init__(self):
        self.count = 0
    def visit_Assert(self, node):
        self.count += 1
        self.generic_visit(node)

def count_asserts(test_code: str) -> int:
    try:
        tree = ast.parse(_dedent(test_code))
    except Exception:
        return 0
    c = _AssertCounter()
    c.visit(tree)
    return c.count

# ======== Transform: check(candidate) -> __collect_asserts(candidate) that appends thunks
class _CheckToCollector(ast.NodeTransformer):
    """
    - Finds def check(candidate): ... and renames to __collect_asserts.
    - Replaces each `assert TEST` with: __ASSERTS__.append(lambda: (TEST))
    """
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
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
                    func=ast.Attribute(
                        value=ast.Name(id="__ASSERTS__", ctx=ast.Load()),
                        attr="append",
                        ctx=ast.Load()
                    ),
                    args=[lam],
                    keywords=[]
                )
                new_body.append(ast.copy_location(ast.Expr(value=call), stmt))
            else:
                new_body.append(stmt)
        node.body = new_body
        return node

def thunkify_tests(test_code: str) -> str:
    """
    Returns Python source that defines __collect_asserts(candidate) which,
    when called, appends thunks for each assert into __ASSERTS__.
    """
    tree = ast.parse(_dedent(test_code))
    tree = _CheckToCollector().visit(tree)
    ast.fix_missing_locations(tree)
    prelude = ast.parse("__ASSERTS__ = globals().get('__ASSERTS__', [])\n")
    module = ast.Module(body=prelude.body + tree.body, type_ignores=[])
    return ast.unparse(module)

# ======== Worker (single example)
def _worker_run(q_out, query: str, completion: str, test_code: str, entry_point_hint: Any, expected_total: int):
    try:
        query = _dedent(query)
        completion = _dedent(completion)
        test_code = _dedent(test_code)
        entry_point_name = resolve_entry_point(entry_point_hint, completion)  # string or None

        # Build test thunks source
        try:
            thunk_src = thunkify_tests(test_code)
        except Exception as e:
            # If tests can't be parsed, all fail deterministically
            q_out.put({
                "passed": 0,
                "failed": expected_total,
                "total": expected_total,
                "ok": False,
                "error_type": "tests_parse",
                "error_msg": f"{type(e).__name__}: {e}",
            })
            return

        cand_src_literal = json.dumps(completion)

        full = f"""{PROMPT_IMPORTS}

__ASSERTS__ = []
__CANDIDATE_LOAD_ERR__ = ""
__TOTAL_EXPECTED__ = int({expected_total})

def __stub__(*args, **kwargs):
    raise NotImplementedError("entry point is missing or invalid")

def __get_entry_point():
    if __CANDIDATE_LOAD_ERR__:
        return __stub__
    try:
        return {entry_point_name}
    except Exception:
        return __stub__

# --- load candidate safely (won't abort the module if it has syntax errors) ---
__CANDIDATE_SRC__ = {cand_src_literal}
try:
    exec(__CANDIDATE_SRC__, globals(), globals())
except Exception as e:
    __CANDIDATE_LOAD_ERR__ = f"candidate_syntax_or_exec: {{type(e).__name__}}: {{e}}"

# --- materialize collector & build thunks ---
{thunk_src}

try:
    __collect_asserts(__get_entry_point())
except Exception:
    # even if this fails partway, we'll still use whatever thunks were appended
    pass

# --- run thunks one by one; failures/exceptions counted; never aborts early ---
__PASSED__, __FAILED__ = 0, 0
for _th in list(__ASSERTS__):
    try:
        ok = bool(_th())
    except Exception:
        __FAILED__ += 1
    else:
        if ok:
            __PASSED__ += 1
        else:
            __FAILED__ += 1

# If the collector couldn't append all thunks (e.g., it crashed mid-way),
# count the remainder as failed so totals match the deterministic expected count.
__EXECUTED__ = len(__ASSERTS__)
__REMAINDER__ = max(0, __TOTAL_EXPECTED__ - __EXECUTED__)
__FAILED__ += __REMAINDER__

__TOTAL__ = __TOTAL_EXPECTED__
"""
        g: Dict[str, Any] = {}
        code = compile(full, "<eval_thunks>", "exec")
        exec(code, g, g)

        total  = int(g.get("__TOTAL__", expected_total))
        passed = int(g.get("__PASSED__", 0))
        failed = int(g.get("__FAILED__", total - passed))

        # "ok" only if all asserts passed, candidate loaded, and total>0
        ok = (failed == 0 and total > 0 and not g.get("__CANDIDATE_LOAD_ERR__"))

        res = {
            "passed": passed,
            "failed": failed,
            "total": total,
            "ok": ok if total > 0 else False,
            "error_type": "" if ok else ("candidate" if g.get("__CANDIDATE_LOAD_ERR__") else ("no_asserts" if total == 0 else "assert")),
            "error_msg": g.get("__CANDIDATE_LOAD_ERR__", ""),
        }
    except Exception as e:
        res = {"passed": 0, "failed": expected_total, "total": expected_total, "ok": False, "error_type": "exec", "error_msg": str(e)}

    q_out.put(res)

def eval_with_timeout(query: str, completion: str, test_code: str, entry_point: Any, timeout: int | None = None) -> Dict[str, Any]:
    """
    Thunk-based evaluator with a hard timeout in a subprocess.
    Deterministic totals from AST assert count.
    """
    if timeout is None:
        timeout = TIME_LIMIT_SEC
    expected_total = count_asserts(test_code)
    # ctx = mp.get_context("spawn")
    ctx = mp.get_context("fork")

    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_worker_run, args=(q, query, completion, test_code, entry_point, expected_total))
    p.start()
    p.join(timeout)
    if p.is_alive():
        try:
            p.terminate()
        finally:
            p.join(1)
        # deterministic result on timeout
        return {"passed": 0, "failed": expected_total, "total": expected_total, "ok": False, "error_type": "timeout", "error_msg": f">{timeout}s"}
    try:
        return q.get_nowait()
    except Exception:
        # deterministic result on crash
        return {"passed": 0, "failed": expected_total, "total": expected_total, "ok": False, "error_type": "exec", "error_msg": "no result"}

# ======== Backward-compatible evaluate_code (now thunked + timeout)
def evaluate_code(prompt: str, completion: str, test_code: str, entry_point: Any) -> Dict[str, Any]:
    """
    Compatibility wrapper: runs the thunked evaluator with TIME_LIMIT_SEC.
    Returns dict(passed, failed, total, ok, error_type, error_msg) with totals based on assert-count.
    """
    return eval_with_timeout(prompt, completion, test_code, entry_point, timeout=TIME_LIMIT_SEC)

# ======== NEW: Assert extraction and single-assert runner with stdout capture

import io
import contextlib
import traceback

class _AssertExtractor(ast.NodeVisitor):
    def __init__(self, src: str):
        self.src = src.splitlines()
        self.assert_nodes = []
    def visit_Assert(self, node):
        self.assert_nodes.append(node)
        self.generic_visit(node)

def extract_assert_texts(test_code: str):
    """
    Return list of source-text (best-effort) for each assert in test_code.
    """
    test_code = _dedent(test_code)
    try:
        tree = ast.parse(test_code)
    except Exception:
        return []
    ex = _AssertExtractor(test_code)
    ex.visit(tree)
    texts = []
    for n in ex.assert_nodes:
        try:
            # Python 3.8+: ast.get_source_segment if available
            seg = ast.get_source_segment(test_code, n)
            if not seg:
                seg = f"assert <unparsed at line {getattr(n, 'lineno', '?')}>"
        except Exception:
            seg = f"assert <unknown at line {getattr(n, 'lineno', '?')}>"
        texts.append(seg.strip())
    return texts

def eval_all_asserts_with_output(query: str, completion: str, test_code: str, entry_point: Any, timeout: int | None = None):
    """
    Runs *all* assert thunks individually and returns a dict:
      {
        "total": N,
        "results": [
           {"index": i, "ok": True/False, "stdout": "...", "error_type": "", "error_msg": ""},
           ...
        ]
      }
    """
    if timeout is None:
        timeout = TIME_LIMIT_SEC

    expected_total = count_asserts(test_code)
    if expected_total <= 0:
        return {"total": 0, "results": []}

    # We'll adapt _worker_run to execute one-by-one with stdout capture.
    def _inner():
        local = {}
        out = {"total": expected_total, "results": []}
        # Build full program including candidate + thunk collector
        q = _dedent(query); comp = _dedent(completion); tests = _dedent(test_code)
        entry_point_name = resolve_entry_point(entry_point, comp)

        thunk_src = thunkify_tests(tests)  # may raise; caller wrapped

        cand_src_literal = json.dumps(comp)
        full = f"""{PROMPT_IMPORTS}

__ASSERTS__ = []
__CANDIDATE_LOAD_ERR__ = ""
def __stub__(*args, **kwargs):
    raise NotImplementedError("entry point is missing or invalid")
def __get_entry_point():
    if __CANDIDATE_LOAD_ERR__:
        return __stub__
    try:
        return {entry_point_name}
    except Exception:
        return __stub__

__CANDIDATE_SRC__ = {cand_src_literal}
try:
    exec(__CANDIDATE_SRC__, globals(), globals())
except Exception as e:
    __CANDIDATE_LOAD_ERR__ = f"candidate_syntax_or_exec: {{type(e).__name__}}: {{e}}"

{thunk_src}

try:
    __collect_asserts(__get_entry_point())
except Exception:
    pass
"""
        g: Dict[str, Any] = {}
        code = compile(full, "<eval_all_asserts>", "exec")
        exec(code, g, g)

        thunks = list(g.get("__ASSERTS__", []))
        for idx, th in enumerate(thunks):
            buf = io.StringIO()
            ok = False
            err_type = ""
            err_msg = ""
            with contextlib.redirect_stdout(buf):
                try:
                    ok = bool(th())
                except Exception as e:
                    ok = False
                    err_type = type(e).__name__
                    err_msg = str(e)
            out["results"].append({
                "index": idx,
                "ok": bool(ok),
                "stdout": buf.getvalue(),
                "error_type": err_type,
                "error_msg": err_msg[:2000],
            })
        return out

    # Run _inner in a subprocess with timeout
    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=1)
    def _proc(q_):
        try:
            res = _inner()
        except Exception as e:
            res = {"total": expected_total, "results": [], "error": f"{type(e).__name__}: {e}"}
        q_.put(res)

    p = ctx.Process(target=_proc, args=(q,))
    p.start()
    p.join(timeout)
    if p.is_alive():
        try: p.terminate()
        finally: p.join(1)
        return {"total": expected_total, "results": [], "error": f"timeout >{timeout}s"}
    try:
        return q.get_nowait()
    except Exception:
        return {"total": expected_total, "results": [], "error": "no result"}

def eval_single_assert_with_output(query: str, completion: str, test_code: str, entry_point: Any, assert_index: int, timeout: int | None = None):
    """
    Execute exactly one assert thunk by index, capture stdout, return:
      {"ok": bool, "stdout": str, "error_type": str, "error_msg": str}
    """
    allres = eval_all_asserts_with_output(query, completion, test_code, entry_point, timeout=timeout)
    N = int(allres.get("total", 0))
    if not allres.get("results"):
        return {"ok": False, "stdout": "", "error_type": "exec", "error_msg": str(allres.get("error", "no thunks"))}
    if assert_index < 0 or assert_index >= N:
        return {"ok": False, "stdout": "", "error_type": "index", "error_msg": f"assert_index {assert_index} out of range [0,{N-1}]"}
    r = allres["results"][assert_index]
    return {"ok": bool(r.get("ok")), "stdout": r.get("stdout", ""), "error_type": r.get("error_type",""), "error_msg": r.get("error_msg","")}

