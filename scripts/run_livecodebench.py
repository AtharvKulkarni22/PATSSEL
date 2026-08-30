#!/usr/bin/env python3
"""
Unified LiveCodeBench repair/generate/test runner.

This is the LiveCodeBench analogue of hefix_all_runs_with_inference_pipeline_updated.py.
It keeps the same core experiment families:
  - buggy accuracy
  - generation only
  - debug only
  - oracle routing
  - generated-test verifier routing using base/RL testgen tests
  - non-backtracking
  - oracle-test backtracking
  - RL-generated-test backtracking

LCB adaptation notes:
  - Inputs are stdin/stdout programs, not LeetCode/HE+Fix callable functions.
  - Buggy programs come from the committed data/livecodebench/test.jsonl bug farm.
  - Each combined bug row keeps bug_source_model_tag/model-id fields, so every buggy row
    can be traced to qwen3_1p7b/qwen3_4b/qwen3_8b/qwen3_14b.
  - Generated verifier tests are stdin/stdout JSON tests from per_problem_generated_tests.jsonl.
  - Final reported metrics are computed on LCB official/public+private tests loaded locally.

Typical full run:
  python scripts/run_livecodebench.py \
    --all \
    --log_dir outputs/results/livecodebench \
    --base_model_id Qwen/Qwen3-8B \
    --rl_peft_checkpoint checkpoints/debugger \
    --base_testgen_jsonl outputs/verifier_tests/livecodebench/base/per_problem_generated_tests.jsonl \
    --rl_testgen_jsonl outputs/verifier_tests/livecodebench/rl/per_problem_generated_tests.jsonl
"""

from __future__ import annotations

# Make local LiveCodeBench package importable when this file is run from scripts/.
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import dataclasses
import json
import random
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from patssel.evaluation.livecodebench import (
    ensure_text,
    normalize_stdout,
    parse_test_cases,
    load_lcb_problem_bank,
    evaluate_code_on_tests,
)

DEFAULT_SEED = 1012
DEFAULT_RELEASE_VERSION = "release_v1"
DEFAULT_CODEGEN_ROOT = str(_REPO_ROOT / "data/livecodebench/test.jsonl")
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_RL_PEFT_CHECKPOINT = str(_REPO_ROOT / "checkpoints/debugger")
DEFAULT_BASE_TESTGEN_JSONL = str(_REPO_ROOT / "outputs/verifier_tests/livecodebench/base/per_problem_generated_tests.jsonl")
DEFAULT_RL_TESTGEN_JSONL = str(_REPO_ROOT / "outputs/verifier_tests/livecodebench/rl/per_problem_generated_tests.jsonl")
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_TIMEOUT = 6.0

FENCE_PY_RE = re.compile(r"```(?:python|py)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
GENERIC_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+\-]*\s*\n([\s\S]*?)\n```$", re.S)

MODEL_TAG_TO_ID = {
    "qwen3_1p7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
}

# -------------------------
# Generic helpers
# -------------------------
def expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(p or "")))


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_jsonl_line(path: str | Path, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def strip_code_fences(text: str) -> str:
    """Extract executable Python from model output.

    Handles:
      - Qwen <think>...</think> or dangling </think>
      - explanation before FINAL FIX
      - closed fenced code blocks
      - unclosed final ```python fences caused by max_new_tokens truncation
      - unfenced Python fallback
    """
    if not text:
        return ""

    t = str(text).strip()

    # Remove complete thinking blocks.
    t = re.sub(r"(?is)<think>.*?</think>", "", t).strip()

    # If generation contains only a dangling closing think tag, keep text after it.
    if "</think>" in t.lower():
        parts = re.split(r"(?is)</think>", t)
        t = parts[-1].strip()

    # Prefer text after final-answer markers.
    marked_t = t
    marker_patterns = [
        r"(?is)4\)\s*FINAL\s+FIX\s*:?",
        r"(?is)FINAL\s+FIX\s*:?",
        r"(?is)FINAL\s+CODE\s*:?",
        r"(?is)CORRECTED\s+CODE\s*:?",
        r"(?is)###\s*Answer\s*:?",
        r"(?is)###\s*Final\s+Answer\s*:?",
    ]
    for pat in marker_patterns:
        ms = list(re.finditer(pat, t))
        if ms:
            marked_t = t[ms[-1].end():].strip()
            break

    # Closed Python fenced blocks after marker.
    matches = FENCE_PY_RE.findall(marked_t)
    if matches:
        return matches[-1].strip()

    # Closed generic fenced blocks after marker.
    generic_blocks = re.findall(r"```[a-zA-Z0-9_+\-]*\s*\n([\s\S]*?)\n```", marked_t)
    if generic_blocks:
        return generic_blocks[-1].strip()

    # Unclosed final python fence after marker.
    m = list(re.finditer(r"```(?:python|py)?\s*\n", marked_t, flags=re.IGNORECASE))
    if m:
        return marked_t[m[-1].end():].strip().rstrip("`").strip()

    # Closed fenced blocks in full output.
    matches = FENCE_PY_RE.findall(t)
    if matches:
        return matches[-1].strip()

    generic_blocks = re.findall(r"```[a-zA-Z0-9_+\-]*\s*\n([\s\S]*?)\n```", t)
    if generic_blocks:
        return generic_blocks[-1].strip()

    # Unclosed python fence in full output.
    m = list(re.finditer(r"```(?:python|py)?\s*\n", t, flags=re.IGNORECASE))
    if m:
        return t[m[-1].end():].strip().rstrip("`").strip()

    # Last-resort: start at first likely Python line.
    x = marked_t.strip()
    lines = x.splitlines()
    starters = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__",
        "sys.",
        "input",
        "t = ",
        "n = ",
        "for ",
        "while ",
    )
    for i, line in enumerate(lines):
        if line.lstrip().startswith(starters):
            return "\n".join(lines[i:]).strip()

    return x


def extract_generation_code(text: str) -> str:
    return strip_code_fences(text)






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


def json_safe_public_eval(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not d:
        return None
    drop = {"details", "status", "exec_out"}
    return {k: v for k, v in d.items() if k not in drop}


def clip_middle(x: Any, max_chars: int, label: str = "text") -> str:
    """Keep prompts bounded while preserving beginning and end context."""
    t = ensure_text(x)
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    head = max_chars // 2
    tail = max_chars - head
    return (
        t[:head]
        + f"\n\n[... {label} truncated: {len(t)} chars total, kept {head}+{tail} chars ...]\n\n"
        + t[-tail:]
    )


def compact_problem(problem: Dict[str, Any]) -> Dict[str, str]:
    return {
        "question_title": clip_middle(problem.get("question_title", ""), 500, "title"),
        "question_id": clip_middle(problem.get("question_id", ""), 200, "question_id"),
        "platform": clip_middle(problem.get("platform", ""), 200, "platform"),
        "difficulty": clip_middle(problem.get("difficulty", ""), 200, "difficulty"),
        "question_content": clip_middle(problem.get("question_content", ""), 10000, "problem statement"),
        "starter_code": clip_middle(problem.get("starter_code", "") or "", 2500, "starter code"),
    }


def compact_target(target: Dict[str, Any]) -> Dict[str, str]:
    return {
        "input": clip_middle(target.get("input", ""), 2500, "target input"),
        "expected": clip_middle(target.get("expected", ""), 2500, "target expected output"),
        "got": clip_middle(target.get("got", ""), 6000, "buggy stdout"),
        "stderr": clip_middle(target.get("stderr", ""), 4000, "buggy stderr"),
    }



# -------------------------
# LCB dataset and bug/test loading
# -------------------------





def load_bug_farms(codegen_root: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load the committed combined bug farm or the original per-model bug-farm tree."""
    root = Path(expand_path(codegen_root))
    if root.is_file():
        rows = load_jsonl(root)
        if not rows:
            raise FileNotFoundError(f"No bug-farm rows found in {root}")
        return rows, str(root)

    paths = sorted(root.glob("*/*_bug_farm.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No *_bug_farm.jsonl files found under {root}")
    rows: List[Dict[str, Any]] = []
    for p in paths:
        default_tag = p.parent.name
        for rec in load_jsonl(p):
            tag = str(rec.get("base_model_tag", default_tag) or default_tag)
            qid = str(rec.get("question_id", ""))
            code_idx = int(rec.get("code_idx", 0) or 0)
            row = dict(rec)
            row["base_model_tag"] = tag
            row["bug_source_model_tag"] = tag
            row["bug_source_model_id"] = MODEL_TAG_TO_ID.get(tag, tag)
            row["bug_source_file"] = str(p)
            row["bug_uid"] = f"{tag}::{qid}::{code_idx}::{len(rows)}"
            row["task_id"] = row["bug_uid"]
            rows.append(row)
    return rows, str(root)


def load_verifier_tests_map(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    jsonl_path = expand_path(jsonl_path)
    if not jsonl_path or not os.path.exists(jsonl_path):
        return out
    for rec in load_jsonl(jsonl_path):
        qid = str(rec.get("question_id", "")).strip()
        if not qid:
            continue
        tests = rec.get("valid_generated_tests") or rec.get("valid_tests") or rec.get("tests") or []
        tests = parse_test_cases(tests)
        out[qid] = {
            "verifier_tests": tests,
            "num_tests": len(tests),
            "record": rec,
        }
    return out

# -------------------------
# Stdin/stdout execution
# -------------------------




def choose_candidate(debug_eval: Dict[str, Any], gen_eval: Dict[str, Any]) -> str:
    if gen_eval["full_suite_passed"] > debug_eval["full_suite_passed"]:
        return "generation"
    if gen_eval["full_suite_passed"] < debug_eval["full_suite_passed"]:
        return "debug"
    if gen_eval.get("problem_passed") and not debug_eval.get("problem_passed"):
        return "generation"
    if debug_eval.get("problem_passed") and not gen_eval.get("problem_passed"):
        return "debug"
    if gen_eval.get("target_ok") and not debug_eval.get("target_ok"):
        return "generation"
    if debug_eval.get("target_ok") and not gen_eval.get("target_ok"):
        return "debug"
    return "generation"


def choose_best_candidate(candidate_evals: List[Dict[str, Any]]) -> int:
    if not candidate_evals:
        raise ValueError("candidate_evals must be non-empty")
    best_idx = 0
    best_key = (
        int(candidate_evals[0].get("full_suite_passed", 0) or 0),
        int(bool(candidate_evals[0].get("problem_passed", False))),
        int(bool(candidate_evals[0].get("target_ok", False))),
    )
    for i, ev in enumerate(candidate_evals[1:], start=1):
        key = (
            int(ev.get("full_suite_passed", 0) or 0),
            int(bool(ev.get("problem_passed", False))),
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

# -------------------------
# Prompt builders
# -------------------------
def build_debug_messages(problem: Dict[str, Any], buggy_code: str, target: Dict[str, Any]) -> List[Dict[str, str]]:
    problem = compact_problem(problem)
    target = compact_target(target)
    buggy_code = clip_middle(buggy_code, 6000, "buggy program")

    system_msg = (
        "You are a Python competitive-programming bug-fixing assistant.\n"
        "Return ONLY the corrected complete Python 3 stdin/stdout program.\n"
        "Put the program in exactly one Python fenced code block.\n"
        "Do not include CODE INTENT, BUG REVIEW, FIX PLAN, explanations, markdown outside the code block, or comments about your reasoning.\n"
    )

    user_msg = (
        "Fix the buggy program so it solves the programming problem.\n"
        "Return a complete Python 3 program that reads from stdin and writes to stdout.\n\n"
        f"Problem title: {problem.get('question_title', '')}\n"
        f"Question id: {problem.get('question_id', '')}\n"
        f"Platform: {problem.get('platform', '')}\n"
        f"Difficulty: {problem.get('difficulty', '')}\n\n"
        "Problem description:\n"
        f"{problem.get('question_content', '')}\n\n"
        "Starter code:\n"
        f"{problem.get('starter_code', '') or '(none)'}\n\n"
        "Buggy program:\n"
        f"```python\n{buggy_code}\n```\n\n"
        "Targeted failing test from the official benchmark:\n"
        f"Input:\n{target.get('input', '')}\n\n"
        f"Expected stdout:\n{target.get('expected', '')}\n\n"
        f"Buggy stdout/stderr:\n{target.get('got', '')}\n{target.get('stderr', '')}\n\n"
        "Output ONLY the corrected full Python program in one ```python fenced code block."
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def build_generation_messages(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    problem = compact_problem(problem)
    system_msg = (
        "You are an expert Python competitive programmer.\n"
        "Return exactly one Python fenced code block containing a complete Python 3 stdin/stdout program.\n"
        "Do not include explanations outside the code block."
    )
    user_msg = (
        "Generate a correct Python 3 solution for this LiveCodeBench problem.\n"
        "The program must read from stdin and write to stdout.\n\n"
        f"Problem title: {problem.get('question_title', '')}\n"
        f"Question id: {problem.get('question_id', '')}\n"
        f"Platform: {problem.get('platform', '')}\n"
        f"Difficulty: {problem.get('difficulty', '')}\n\n"
        "Problem description:\n"
        f"{problem.get('question_content', '')}\n\n"
        "Starter code:\n"
        f"{problem.get('starter_code', '') or '(none)'}\n"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def _format_lcb_execution_feedback(item: Dict[str, Any]) -> str:
    """Create LiveCodeBench-style self-repair feedback from stored execution metadata and target failure."""
    meta = item.get("bug_metadata")
    pieces: List[str] = []
    if meta is not None:
        try:
            pieces.append("Previous execution metadata:\n" + clip_middle(json.dumps(meta, ensure_ascii=False, indent=2, default=str), 4000, "execution metadata"))
        except Exception:
            pieces.append("Previous execution metadata:\n" + clip_middle(str(meta), 4000, "execution metadata"))
    target = compact_target(item.get("target", {}) or {})
    if target:
        pieces.append(
            "Representative failing test from the official evaluator:\n"
            f"Input:\n{target.get('input','')}\n\n"
            f"Expected stdout:\n{target.get('expected','')}\n\n"
            f"Observed stdout/stderr:\n{target.get('got','')}\n{target.get('stderr','')}"
        )
    return "\n\n".join(x for x in pieces if x.strip()) or "The previous solution failed the evaluator. Please repair it."


def build_lcb_selfrepair_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """Approximate the official LiveCodeBench self-repair prompt style.

    This is intentionally separate from the paper's structured debugger prompt.
    It gives the problem, the previous answer, and execution feedback, then asks
    for one fixed stdin/stdout Python program in a code block.
    """
    problem = compact_problem(item["problem"])
    feedback = _format_lcb_execution_feedback(item)
    buggy_code = clip_middle(item.get("buggy_code", ""), 6000, "previous buggy program")
    system_msg = (
        "You are a helpful programming assistant and an expert Python programmer. "
        "You will be given a programming problem, a previous Python solution that failed, "
        "and execution feedback. Briefly explain what is wrong in 2-3 sentences, then output "
        "the fixed full Python program in exactly one Python code block."
    )
    user_msg = (
        "### Question:\n"
        f"{problem.get('question_content','')}\n\n"
        "### Previous Answer:\n"
        f"```python\n{buggy_code}\n```\n\n"
        "### Execution Feedback:\n"
        f"{feedback}\n\n"
        "### Format:\n"
        "Read input from stdin and write output to stdout. Return a complete Python 3 program.\n"
        "Use the following format and put the fixed code between backticks:\n"
        "```python\n# YOUR CODE HERE\n```\n\n"
        "### Answer:"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

# -------------------------
# Model wrapper
# -------------------------
class ModelRunner:
    def __init__(self, model, tokenizer, model_id: str, peft_checkpoint: Optional[str] = None, prompt_max_tokens: int = 32768, disable_adapter: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.peft_checkpoint = peft_checkpoint
        self.prompt_max_tokens = int(prompt_max_tokens)
        self.disable_adapter = bool(disable_adapter)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def render_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def generate_batch(self, batch_messages: List[List[Dict[str, str]]], deterministic: bool, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> List[Tuple[str, int]]:
        """Generate prompt-by-prompt to avoid OOM on 11GB GPUs."""
        if not batch_messages:
            return []

        results: List[Tuple[str, int]] = []

        for messages in batch_messages:
            prompt = self.render_chat_prompt(messages)
            prompt_len = len(prompt)

            enc = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(self, "prompt_max_tokens", 32768),
            )

            dev = first_shard_device(self.model)
            enc = {k: v.to(dev) for k, v in enc.items()}
            input_len = int(enc["attention_mask"].sum(dim=1).tolist()[0])

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            if deterministic or temperature <= 0:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                if top_k and top_k > 0:
                    gen_kwargs["top_k"] = top_k

            adapter_ctx = (
                self.model.disable_adapter()
                if self.disable_adapter and hasattr(self.model, "disable_adapter")
                else nullcontext()
            )

            with adapter_ctx:
                outputs = self.model.generate(**enc, **gen_kwargs)

            gen_ids = outputs[0][input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append((text, prompt_len))

            try:
                del enc, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

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


def load_model_and_tokenizer(model_id: str, peft_checkpoint: Optional[str], dtype: str, trust_remote_code: bool, prompt_max_tokens: int = 32768) -> ModelRunner:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    torch_dtype = _dtype_from_string(dtype)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=trust_remote_code)
    if peft_checkpoint:
        if PeftModel is None:
            raise RuntimeError("peft is unavailable, but --rl_peft_checkpoint was provided")
        model = PeftModel.from_pretrained(base, peft_checkpoint)
    else:
        model = base
    model.eval()
    return ModelRunner(model, tokenizer, model_id=model_id, peft_checkpoint=peft_checkpoint, prompt_max_tokens=prompt_max_tokens)


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
# Experiment definitions
# -------------------------
@dataclass(frozen=True)
class Experiment:
    name: str
    kind: str  # buggy, single_debug, single_generation, routing_oracle, routing_verifier
    debug_model: Optional[str] = None
    generation_model: Optional[str] = None
    verifier_source: Optional[str] = None
    table_label: str = ""


EXPERIMENTS: Dict[str, Experiment] = {
    # Qwen rows in the paper main table.
    "base_regen": Experiment("base_regen", "best2_generation_verifier", generation_model="base", verifier_source="base_testgen", table_label="Base ReGen"),
    "base_debug": Experiment("base_debug", "best2_debug_verifier", debug_model="rl", verifier_source="base_testgen", table_label="Base Debug"),
    "base_patssel": Experiment("base_patssel", "routing_verifier", debug_model="rl", generation_model="base", verifier_source="base_testgen", table_label="Base PATSSEL"),

    "rl_regen": Experiment("rl_regen", "best2_generation_verifier", generation_model="base", verifier_source="rl_testgen", table_label="RL ReGen"),
    "rl_debug": Experiment("rl_debug", "best2_debug_verifier", debug_model="rl", verifier_source="rl_testgen", table_label="RL Debug"),
    "rl_patssel": Experiment("rl_patssel", "routing_verifier", debug_model="rl", generation_model="base", verifier_source="rl_testgen", table_label="RL PATSSEL"),

    "gold_regen": Experiment("gold_regen", "best2_generation_oracle", generation_model="base", table_label="Gold ReGen"),
    "gold_debug": Experiment("gold_debug", "best2_debug_oracle", debug_model="rl", table_label="Gold Debug"),
    "gold_patssel": Experiment("gold_patssel", "routing_oracle", debug_model="rl", generation_model="base", table_label="Gold PATSSEL"),

    # Optional diagnostics.
    "buggy_acc": Experiment("buggy_acc", "buggy", table_label="Buggy Code Acc"),
    "base_gen_only": Experiment("base_gen_only", "single_generation", generation_model="base", table_label="Base Gen Only"),
    "rl_debug_only": Experiment("rl_debug_only", "single_debug", debug_model="rl", table_label="RL Debug Only"),
}

ALL_EXPERIMENT_ORDER = [
    "base_regen", "base_debug", "base_patssel",
    "rl_regen", "rl_debug", "rl_patssel",
    "gold_regen", "gold_debug", "gold_patssel",
]

# -------------------------
# Row preparation and metrics
# -------------------------
def prepare_row(bug: Dict[str, Any], problem_bank: Dict[str, Dict[str, Any]], args) -> Optional[Dict[str, Any]]:
    qid = str(bug.get("question_id", "")).strip()
    problem = problem_bank.get(qid)
    if not problem:
        return None
    official_tests = problem.get("official_tests") or []
    if not official_tests:
        return None
    buggy_code = str(bug.get("buggy_code", "") or bug.get("buggy_raw_output", "") or "")
    baseline = evaluate_code_on_tests(buggy_code, official_tests, args.timeout, target_idx=0, keep_details=True)
    failing_idxs = [d["index"] for d in baseline.get("details", []) if not d.get("ok")]
    if args.target_policy == "random" and failing_idxs:
        target_idx = int(random.choice(failing_idxs))
    else:
        target_idx = int(failing_idxs[0]) if failing_idxs else -1
    target_detail = next((d for d in baseline.get("details", []) if d.get("index") == target_idx), {}) if target_idx >= 0 else {}
    return {
        "task_id": str(bug.get("task_id", bug.get("bug_uid", qid))),
        "bug_uid": str(bug.get("bug_uid", bug.get("task_id", qid))),
        "question_id": qid,
        "question_title": problem.get("question_title", bug.get("question_title", "")),
        "difficulty": problem.get("difficulty", bug.get("difficulty", "")),
        "platform": problem.get("platform", bug.get("platform", "")),
        "problem": problem,
        "official_tests": official_tests,
        "buggy_code": buggy_code,
        "bug_metadata": bug.get("metadata"),
        "bug_source_model_tag": bug.get("bug_source_model_tag", bug.get("base_model_tag", "")),
        "bug_source_model_id": bug.get("bug_source_model_id", ""),
        "baseline_eval": baseline,
        "target_idx": target_idx,
        "target": {
            "input": target_detail.get("input", ""),
            "expected": target_detail.get("expected", ""),
            "got": target_detail.get("got", ""),
            "stderr": target_detail.get("stderr", ""),
        },
    }


def build_default_result(item: Dict[str, Any], note: str) -> Dict[str, Any]:
    return {
        "selected_source": "buggy",
        "selected_code": item["buggy_code"],
        "nonbacktracking_eval": item["baseline_eval"],
        "backtracking_with_oracle_tests_source": "buggy",
        "backtracking_with_oracle_tests_eval": item["baseline_eval"],
        "backtracking_with_rl_generated_tests_source": "buggy",
        "backtracking_with_rl_generated_tests_eval": item["baseline_eval"],
        "rl_backtracking_baseline_verifier_eval": None,
        "rl_backtracking_selected_verifier_eval": None,
        "note": note,
    }


def compute_oracle_backtracking(item: Dict[str, Any], selected_source: str, selected_eval: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # Keep selected only if it strictly improves official-test pass count over the original buggy code.
    if selected_eval["full_suite_passed"] > item["baseline_eval"]["full_suite_passed"]:
        return selected_source, selected_eval
    return "buggy", item["baseline_eval"]


def compute_rl_generated_backtracking(item: Dict[str, Any], selected_source: str, selected_code: str, selected_eval: Dict[str, Any], rl_tests_map: Dict[str, Dict[str, Any]], args) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if selected_source == "buggy":
        return "buggy", item["baseline_eval"], None, None
    vinfo = rl_tests_map.get(item["question_id"], {})
    tests = vinfo.get("verifier_tests", []) or []
    if not tests:
        return "buggy", item["baseline_eval"], None, None
    base_v = evaluate_code_on_tests(item["buggy_code"], tests, args.timeout, 0, keep_details=False)
    sel_v = evaluate_code_on_tests(selected_code, tests, args.timeout, 0, keep_details=False)
    if sel_v["full_suite_passed"] > base_v["full_suite_passed"]:
        return selected_source, selected_eval, base_v, sel_v
    return "buggy", item["baseline_eval"], base_v, sel_v


def target_fixed_for_eval(item: Dict[str, Any], eval_obj: Optional[Dict[str, Any]]) -> int:
    """LCB targeted-fix metric.

    The target is chosen from the original buggy program's first/random failing
    official LCB test. A candidate gets target_fixed=1 only if it passes that
    same official test index. This is separate from full-suite pass rate.
    """
    if eval_obj is None:
        return 0
    if int(item.get("target_idx", -1)) < 0:
        return 0
    return int(bool(eval_obj.get("target_ok", False)))


def _add_count(d: Dict[str, int], key: str) -> None:
    d[key or "unknown"] = d.get(key or "unknown", 0) + 1


def load_resume_state(summary_jsonl: str, experiment_name: str) -> Dict[str, Any]:
    state = {
        "completed_ids": set(), "evaluated": 0, "skipped": 0,
        "nb_passed": 0, "nb_total": 0, "nb_problem_passed": 0, "nb_target": 0,
        "bto_passed": 0, "bto_total": 0, "bto_problem_passed": 0, "bto_target": 0,
        "btr_passed": 0, "btr_total": 0, "btr_problem_passed": 0, "btr_target": 0,
        "nb_counts": {}, "bto_counts": {}, "btr_counts": {},
    }
    if not os.path.exists(summary_jsonl):
        return state
    by_id: Dict[str, Dict[str, Any]] = {}
    with open(summary_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("experiment") not in (None, "", experiment_name):
                continue
            uid = str(rec.get("bug_uid", rec.get("task_id", ""))).strip()
            if uid:
                by_id[uid] = rec
    state["completed_ids"] = set(by_id.keys())
    for rec in by_id.values():
        if rec.get("event") == "skipped_prepare_row":
            state["skipped"] += 1
            continue
        if rec.get("event") != "final_metrics":
            continue
        state["evaluated"] += 1
        for prefix, key in [("nb", "nonbacktracking_eval"), ("bto", "backtracking_with_oracle_tests_eval"), ("btr", "backtracking_with_rl_generated_tests_eval")]:
            ev = rec.get(key, {}) or {}
            state[f"{prefix}_passed"] += int(ev.get("full_suite_passed", 0) or 0)
            state[f"{prefix}_total"] += int(ev.get("full_suite_total", 0) or 0)
            state[f"{prefix}_problem_passed"] += int(bool(ev.get("problem_passed", False)))
            explicit_key = {
                "nb": "nonbacktracking_target_fixed",
                "bto": "backtracking_with_oracle_tests_target_fixed",
                "btr": "backtracking_with_rl_generated_tests_target_fixed",
            }[prefix]
            state[f"{prefix}_target"] += int(bool(rec.get(explicit_key, ev.get("target_ok", False))))
        _add_count(state["nb_counts"], rec.get("selected_source", "unknown"))
        _add_count(state["bto_counts"], rec.get("backtracking_with_oracle_tests_source", "unknown"))
        _add_count(state["btr_counts"], rec.get("backtracking_with_rl_generated_tests_source", "unknown"))
    return state


def build_final_metrics(experiment: Experiment, mode: str, problems_seen: int, evaluated: int, skipped: int, passed: int, total: int, problem_passed: int, target_fixed: int, source_counts: Dict[str, int], args) -> Dict[str, Any]:
    source_counts_clean = {k: int(v) for k, v in sorted(source_counts.items())}
    return {
        "experiment": experiment.name,
        "table_label": experiment.table_label,
        "kind": experiment.kind,
        "dataset": "LiveCodeBench",
        "release_version": args.release_version,
        "mode": mode,
        "problems_seen": int(problems_seen),
        "evaluated": int(evaluated),
        "skipped_count": int(skipped),
        "final_full_suite_passed_total": int(passed),
        "final_full_suite_total_tests": int(total),
        "overall_test_accuracy_pct": float(100.0 * passed / max(1, total)) if total else 0.0,
        "problem_passed_count": int(problem_passed),
        "problem_pass_rate_pct": float(100.0 * problem_passed / max(1, evaluated)) if evaluated else 0.0,
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
        "timeout": float(args.timeout),
    }

# -------------------------
# Core runner
# -------------------------
def needed_models(experiments: List[Experiment]) -> Tuple[bool, bool]:
    need_base = any(e.debug_model == "base" or e.generation_model == "base" for e in experiments)
    need_rl = any(e.debug_model == "rl" or e.generation_model == "rl" for e in experiments)
    return need_base, need_rl


def run_experiment(experiment: Experiment, args, rows: List[Dict[str, Any]], problem_bank: Dict[str, Dict[str, Any]], runners: Dict[str, ModelRunner], verifier_maps: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    root = Path(args.log_dir) / experiment.name
    out_non = root / "non_backtracking"
    out_bt_oracle = root / "backtracking_with_oracle_tests"
    out_bt_rl = root / "backtracking_with_rl_generated_tests"
    ensure_dir(out_non); ensure_dir(out_bt_oracle); ensure_dir(out_bt_rl)

    final_non_path = out_non / "final_metrics.json"
    final_bt_oracle_path = out_bt_oracle / "final_metrics.json"
    final_bt_rl_path = out_bt_rl / "final_metrics.json"
    final_all_path = root / "final_metrics_all.json"

    if not args.force and final_non_path.exists() and final_bt_oracle_path.exists() and final_bt_rl_path.exists():
        print(f"[cache] {experiment.name}: loading final metrics")
        return json.load(open(final_all_path, "r", encoding="utf-8")) if final_all_path.exists() else {}

    attempts_tsv = out_non / "attempts_log_with_full_code_and_suite.tsv"
    baseline_tsv = out_non / "baseline_metrics.tsv"
    summary_jsonl = out_non / "summary_per_problem.jsonl"

    if args.force:
        for p in [attempts_tsv, baseline_tsv, summary_jsonl, final_non_path, final_bt_oracle_path, final_bt_rl_path, final_all_path]:
            Path(p).unlink(missing_ok=True)

    resume = load_resume_state(str(summary_jsonl), experiment.name) if not args.force else load_resume_state("/no/such/path", experiment.name)
    completed_ids = set(resume["completed_ids"])
    if completed_ids:
        print(f"[resume] {experiment.name}: {len(completed_ids)} completed bug rows")

    attempts_header = [
        "experiment", "bug_uid", "question_id", "bug_source_model_tag", "target_idx",
        "baseline_passed", "baseline_total", "baseline_acc_pct", "baseline_problem_passed",
        "debug_passed", "debug_total", "debug_acc_pct", "debug_problem_passed",
        "generation_passed", "generation_total", "generation_acc_pct", "generation_problem_passed",
        "verifier_debug_passed", "verifier_debug_total", "verifier_generation_passed", "verifier_generation_total",
        "selected_source", "nonbacktracking_passed", "nonbacktracking_total", "nonbacktracking_acc_pct", "nonbacktracking_problem_passed",
        "backtracking_with_oracle_tests_source", "backtracking_with_oracle_tests_passed", "backtracking_with_oracle_tests_total", "backtracking_with_oracle_tests_acc_pct", "backtracking_with_oracle_tests_problem_passed",
        "backtracking_with_rl_generated_tests_source", "backtracking_with_rl_generated_tests_passed", "backtracking_with_rl_generated_tests_total", "backtracking_with_rl_generated_tests_acc_pct", "backtracking_with_rl_generated_tests_problem_passed",
        "rl_backtracking_baseline_verifier_passed", "rl_backtracking_baseline_verifier_total",
        "rl_backtracking_selected_verifier_passed", "rl_backtracking_selected_verifier_total",
        "target_ok", "debug_prompt_chars", "generation_prompt_chars", "debug_output_chars", "generation_output_chars",
        "target_input", "target_expected", "target_got_baseline", "target_got_selected", "note",
        "debug_code_full", "generation_code_full", "selected_code_full",
    ]
    baseline_header = ["bug_uid", "question_id", "bug_source_model_tag", "baseline_passed", "baseline_total", "baseline_acc_pct", "baseline_problem_passed", "has_failing_test", "target_idx", "note"]

    def open_tsv(path: Path, header: List[str]):
        need_header = (not path.exists()) or path.stat().st_size == 0
        fp = path.open("a", newline="", encoding="utf-8")
        w = csv.writer(fp, delimiter="\t")
        if need_header:
            w.writerow(header); fp.flush()
        return fp, w

    attempts_fp, attempts_writer = open_tsv(attempts_tsv, attempts_header)
    baseline_fp, baseline_writer = open_tsv(baseline_tsv, baseline_header)

    nb_passed = int(resume["nb_passed"]); nb_total = int(resume["nb_total"]); nb_problem = int(resume["nb_problem_passed"]); nb_target = int(resume["nb_target"])
    bto_passed = int(resume["bto_passed"]); bto_total = int(resume["bto_total"]); bto_problem = int(resume["bto_problem_passed"]); bto_target = int(resume["bto_target"])
    btr_passed = int(resume["btr_passed"]); btr_total = int(resume["btr_total"]); btr_problem = int(resume["btr_problem_passed"]); btr_target = int(resume["btr_target"])
    nb_counts = dict(resume["nb_counts"]); bto_counts = dict(resume["bto_counts"]); btr_counts = dict(resume["btr_counts"])
    evaluated = int(resume["evaluated"]); skipped = int(resume["skipped"])
    pending: List[Dict[str, Any]] = []

    def write_result(item: Dict[str, Any], result: Dict[str, Any]):
        nonlocal nb_passed, nb_total, nb_problem, nb_target, bto_passed, bto_total, bto_problem, bto_target, btr_passed, btr_total, btr_problem, btr_target, evaluated
        evaluated += 1
        nb_eval = result["nonbacktracking_eval"]
        bto_eval = result["backtracking_with_oracle_tests_eval"]
        btr_eval = result["backtracking_with_rl_generated_tests_eval"]
        for ev, prefix in [(nb_eval, "nb"), (bto_eval, "bto"), (btr_eval, "btr")]:
            pass
        nb_target_fixed = target_fixed_for_eval(item, nb_eval)
        bto_target_fixed = target_fixed_for_eval(item, bto_eval)
        btr_target_fixed = target_fixed_for_eval(item, btr_eval)

        nb_passed += int(nb_eval["full_suite_passed"]); nb_total += int(nb_eval["full_suite_total"]); nb_problem += int(bool(nb_eval.get("problem_passed"))); nb_target += nb_target_fixed
        bto_passed += int(bto_eval["full_suite_passed"]); bto_total += int(bto_eval["full_suite_total"]); bto_problem += int(bool(bto_eval.get("problem_passed"))); bto_target += bto_target_fixed
        btr_passed += int(btr_eval["full_suite_passed"]); btr_total += int(btr_eval["full_suite_total"]); btr_problem += int(bool(btr_eval.get("problem_passed"))); btr_target += btr_target_fixed
        _add_count(nb_counts, result["selected_source"]); _add_count(bto_counts, result["backtracking_with_oracle_tests_source"]); _add_count(btr_counts, result["backtracking_with_rl_generated_tests_source"])

        debug_eval = result.get("debug_eval") or {}
        gen_eval = result.get("generation_eval") or {}
        debug_v = result.get("debug_verifier_eval") or {}
        gen_v = result.get("generation_verifier_eval") or {}
        rl_base_v = result.get("rl_backtracking_baseline_verifier_eval") or {}
        rl_sel_v = result.get("rl_backtracking_selected_verifier_eval") or {}
        attempts_writer.writerow([
            experiment.name, item["bug_uid"], item["question_id"], item.get("bug_source_model_tag", ""), item["target_idx"],
            item["baseline_eval"]["full_suite_passed"], item["baseline_eval"]["full_suite_total"], f"{item['baseline_eval']['full_suite_acc_pct']:.2f}", int(bool(item["baseline_eval"].get("problem_passed"))),
            debug_eval.get("full_suite_passed", ""), debug_eval.get("full_suite_total", ""), f"{debug_eval.get('full_suite_acc_pct', 0.0):.2f}" if debug_eval else "", int(bool(debug_eval.get("problem_passed", False))) if debug_eval else "",
            gen_eval.get("full_suite_passed", ""), gen_eval.get("full_suite_total", ""), f"{gen_eval.get('full_suite_acc_pct', 0.0):.2f}" if gen_eval else "", int(bool(gen_eval.get("problem_passed", False))) if gen_eval else "",
            debug_v.get("full_suite_passed", ""), debug_v.get("full_suite_total", ""), gen_v.get("full_suite_passed", ""), gen_v.get("full_suite_total", ""),
            result["selected_source"], nb_eval["full_suite_passed"], nb_eval["full_suite_total"], f"{nb_eval['full_suite_acc_pct']:.2f}", int(bool(nb_eval.get("problem_passed"))),
            result["backtracking_with_oracle_tests_source"], bto_eval["full_suite_passed"], bto_eval["full_suite_total"], f"{bto_eval['full_suite_acc_pct']:.2f}", int(bool(bto_eval.get("problem_passed"))),
            result["backtracking_with_rl_generated_tests_source"], btr_eval["full_suite_passed"], btr_eval["full_suite_total"], f"{btr_eval['full_suite_acc_pct']:.2f}", int(bool(btr_eval.get("problem_passed"))),
            rl_base_v.get("full_suite_passed", ""), rl_base_v.get("full_suite_total", ""), rl_sel_v.get("full_suite_passed", ""), rl_sel_v.get("full_suite_total", ""),
            nb_target_fixed, result.get("debug_prompt_chars", 0), result.get("generation_prompt_chars", 0), result.get("debug_output_chars", 0), result.get("generation_output_chars", 0),
            repr(item.get("target", {}).get("input", "")), repr(item.get("target", {}).get("expected", "")), repr(item.get("target", {}).get("got", "")), repr(nb_eval.get("target_got")), result.get("note", ""),
            result.get("debug_code", ""), result.get("generation_code", ""), result.get("selected_code", ""),
        ])
        attempts_fp.flush()
        save_jsonl_line(summary_jsonl, {
            "event": "final_metrics",
            "experiment": experiment.name,
            "task_id": item["task_id"],
            "bug_uid": item["bug_uid"],
            "question_id": item["question_id"],
            "bug_source_model_tag": item.get("bug_source_model_tag", ""),
            "bug_source_model_id": item.get("bug_source_model_id", ""),
            "target_idx": item["target_idx"],
            "baseline_eval": json_safe_public_eval(item["baseline_eval"]),
            "selected_source": result["selected_source"],
            "nonbacktracking_eval": json_safe_public_eval(nb_eval),
            "nonbacktracking_target_fixed": nb_target_fixed,
            "backtracking_with_oracle_tests_source": result["backtracking_with_oracle_tests_source"],
            "backtracking_with_oracle_tests_eval": json_safe_public_eval(bto_eval),
            "backtracking_with_oracle_tests_target_fixed": bto_target_fixed,
            "backtracking_with_rl_generated_tests_source": result["backtracking_with_rl_generated_tests_source"],
            "backtracking_with_rl_generated_tests_eval": json_safe_public_eval(btr_eval),
            "backtracking_with_rl_generated_tests_target_fixed": btr_target_fixed,
            "debug_eval": result.get("debug_eval"),
            "generation_eval": result.get("generation_eval"),
            "debug_verifier_eval": json_safe_public_eval(result.get("debug_verifier_eval")),
            "generation_verifier_eval": json_safe_public_eval(result.get("generation_verifier_eval")),
            "rl_backtracking_baseline_verifier_eval": json_safe_public_eval(result.get("rl_backtracking_baseline_verifier_eval")),
            "rl_backtracking_selected_verifier_eval": json_safe_public_eval(result.get("rl_backtracking_selected_verifier_eval")),
            "note": result.get("note", ""),
        })
        completed_ids.add(item["bug_uid"])

    def write_skip(uid: str, note: str):
        nonlocal skipped
        skipped += 1
        save_jsonl_line(summary_jsonl, {"event": "skipped_prepare_row", "experiment": experiment.name, "bug_uid": uid, "task_id": uid, "note": note})
        completed_ids.add(uid)

    def flush_pending():
        nonlocal pending
        if not pending:
            return

        need_debug = experiment.kind in (
            "single_debug", "best2_debug_oracle", "best2_debug_verifier",
            "routing_oracle", "routing_verifier", "lcb_selfrepair"
        )
        need_gen = experiment.kind in (
            "single_generation", "best2_generation_oracle", "best2_generation_verifier",
            "routing_oracle", "routing_verifier"
        )
        best2_debug = experiment.kind in ("best2_debug_oracle", "best2_debug_verifier")
        best2_gen = experiment.kind in ("best2_generation_oracle", "best2_generation_verifier")

        debug_outs: List[Tuple[str, int]] = []
        gen_outs: List[Tuple[str, int]] = []

        if need_debug:
            runner = runners[experiment.debug_model]
            if experiment.kind == "lcb_selfrepair":
                base_msgs = [build_lcb_selfrepair_messages(x) for x in pending]
            else:
                base_msgs = [build_debug_messages(x["problem"], x["buggy_code"], x["target"]) for x in pending]
            msgs = []
            if best2_debug:
                for m in base_msgs:
                    msgs.extend([m, m])
            else:
                msgs = base_msgs
            debug_outs = runner.generate_batch(msgs, args.deterministic_test, args.max_new_tokens, args.temperature, args.top_p, args.top_k)

        if need_gen:
            runner = runners[experiment.generation_model]
            base_msgs = [build_generation_messages(x["problem"]) for x in pending]
            msgs = []
            if best2_gen:
                for m in base_msgs:
                    msgs.extend([m, m])
            else:
                msgs = base_msgs
            gen_outs = runner.generate_batch(msgs, args.deterministic_test, args.max_new_tokens, args.temperature, args.top_p, args.top_k)

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

            debug_codes = [extract_generation_code(x) for x in debug_raws]
            gen_codes = [extract_generation_code(x) for x in gen_raws]

            debug_raw = debug_raws[0] if debug_raws else ""
            gen_raw = gen_raws[0] if gen_raws else ""
            debug_code = debug_codes[0] if debug_codes else ""
            gen_code = gen_codes[0] if gen_codes else ""

            debug_eval = evaluate_code_on_tests(debug_code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False) if need_debug else None
            gen_eval = evaluate_code_on_tests(gen_code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False) if need_gen else None

            selected_source = "buggy"; selected_code = item["buggy_code"]; selected_eval = item["baseline_eval"]
            debug_v = None; gen_v = None; note = ""
            candidate_evals = None
            candidate_codes = None
            selected_sample_index = None

            if experiment.kind == "lcb_selfrepair":
                selected_source, selected_code, selected_eval = "selfrepair", debug_code, debug_eval

            elif experiment.kind == "single_debug":
                selected_source, selected_code, selected_eval = "debug", debug_code, debug_eval

            elif experiment.kind == "single_generation":
                selected_source, selected_code, selected_eval = "generation", gen_code, gen_eval

            elif experiment.kind == "best2_generation_oracle":
                gen_evals = [evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False) for code in gen_codes]
                selected_sample_index = choose_best_candidate(gen_evals)
                selected_source = f"generation_sample_{selected_sample_index + 1}"
                selected_code = gen_codes[selected_sample_index]
                selected_eval = gen_evals[selected_sample_index]
                gen_eval = selected_eval
                gen_code = selected_code
                gen_raw = gen_raws[selected_sample_index]
                candidate_evals = gen_evals
                candidate_codes = gen_codes
                note = summarize_candidate_evals("gen", gen_evals, selected_sample_index)

            elif experiment.kind == "best2_debug_oracle":
                debug_evals = [evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False) for code in debug_codes]
                selected_sample_index = choose_best_candidate(debug_evals)
                selected_source = f"debug_sample_{selected_sample_index + 1}"
                selected_code = debug_codes[selected_sample_index]
                selected_eval = debug_evals[selected_sample_index]
                debug_eval = selected_eval
                debug_code = selected_code
                debug_raw = debug_raws[selected_sample_index]
                candidate_evals = debug_evals
                candidate_codes = debug_codes
                note = summarize_candidate_evals("debug", debug_evals, selected_sample_index)

            elif experiment.kind == "best2_generation_verifier":
                vinfo = verifier_maps.get(experiment.verifier_source or "", {}).get(item["question_id"], {})
                tests = vinfo.get("verifier_tests", []) or []

                # If the qid exists but has zero valid generated verifier tests, do NOT
                # fall back to the original buggy program. For a ReGen row, the model's
                # candidate strategy is still regeneration. Use sample 1 as the fallback
                # selector when verifier selection is unavailable.
                if not tests:
                    selected_sample_index = 0
                    selected_source = "generation_sample_1"
                    selected_code = gen_codes[0] if gen_codes else ""
                    selected_eval = evaluate_code_on_tests(
                        selected_code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False
                    )
                    gen_eval = selected_eval
                    gen_code = selected_code
                    gen_raw = gen_raws[0] if gen_raws else ""
                    candidate_evals = [selected_eval]
                    candidate_codes = gen_codes
                    note = "missing_or_empty_verifier_fallback_generation_sample_1"
                else:
                    gen_evals_official = [
                        evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False)
                        for code in gen_codes
                    ]
                    gen_evals_verifier = [
                        evaluate_code_on_tests(code, tests, args.timeout, 0, keep_details=False)
                        for code in gen_codes
                    ]
                    selected_sample_index = choose_best_candidate(gen_evals_verifier)
                    selected_source = f"generation_sample_{selected_sample_index + 1}"
                    selected_code = gen_codes[selected_sample_index]
                    selected_eval = gen_evals_official[selected_sample_index]
                    gen_eval = selected_eval
                    gen_v = gen_evals_verifier[selected_sample_index]
                    gen_code = selected_code
                    gen_raw = gen_raws[selected_sample_index]
                    candidate_evals = gen_evals_official
                    candidate_codes = gen_codes
                    note = (
                        "verifier_selection;"
                        + summarize_candidate_evals("verifier_gen", gen_evals_verifier, selected_sample_index)
                        + ";official_"
                        + summarize_candidate_evals("gen", gen_evals_official, selected_sample_index)
                    )

            elif experiment.kind == "best2_debug_verifier":
                vinfo = verifier_maps.get(experiment.verifier_source or "", {}).get(item["question_id"], {})
                tests = vinfo.get("verifier_tests", []) or []

                # If verifier tests are empty, still evaluate the debugger candidate.
                # Use sample 1 as the fallback selector.
                if not tests:
                    selected_sample_index = 0
                    selected_source = "debug_sample_1"
                    selected_code = debug_codes[0] if debug_codes else ""
                    selected_eval = evaluate_code_on_tests(
                        selected_code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False
                    )
                    debug_eval = selected_eval
                    debug_code = selected_code
                    debug_raw = debug_raws[0] if debug_raws else ""
                    candidate_evals = [selected_eval]
                    candidate_codes = debug_codes
                    note = "missing_or_empty_verifier_fallback_debug_sample_1"
                else:
                    debug_evals_official = [
                        evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False)
                        for code in debug_codes
                    ]
                    debug_evals_verifier = [
                        evaluate_code_on_tests(code, tests, args.timeout, 0, keep_details=False)
                        for code in debug_codes
                    ]
                    selected_sample_index = choose_best_candidate(debug_evals_verifier)
                    selected_source = f"debug_sample_{selected_sample_index + 1}"
                    selected_code = debug_codes[selected_sample_index]
                    selected_eval = debug_evals_official[selected_sample_index]
                    debug_eval = selected_eval
                    debug_v = debug_evals_verifier[selected_sample_index]
                    debug_code = selected_code
                    debug_raw = debug_raws[selected_sample_index]
                    candidate_evals = debug_evals_official
                    candidate_codes = debug_codes
                    note = (
                        "verifier_selection;"
                        + summarize_candidate_evals("verifier_debug", debug_evals_verifier, selected_sample_index)
                        + ";official_"
                        + summarize_candidate_evals("debug", debug_evals_official, selected_sample_index)
                    )

            elif experiment.kind == "routing_oracle":
                debug_evals_official = [
                    evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False)
                    for code in debug_codes
                ]
                gen_evals_official = [
                    evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False)
                    for code in gen_codes
                ]

                best_debug_idx = choose_best_candidate(debug_evals_official)
                best_gen_idx = choose_best_candidate(gen_evals_official)

                debug_eval = debug_evals_official[best_debug_idx]
                gen_eval = gen_evals_official[best_gen_idx]
                debug_code = debug_codes[best_debug_idx]
                gen_code = gen_codes[best_gen_idx]
                debug_raw = debug_raws[best_debug_idx]
                gen_raw = gen_raws[best_gen_idx]

                selected_branch = choose_candidate(debug_eval, gen_eval)
                if selected_branch == "debug":
                    selected_source = f"debug_sample_{best_debug_idx + 1}"
                    selected_code = debug_code
                    selected_eval = debug_eval
                    selected_sample_index = best_debug_idx
                else:
                    selected_source = f"generation_sample_{best_gen_idx + 1}"
                    selected_code = gen_code
                    selected_eval = gen_eval
                    selected_sample_index = best_gen_idx

                candidate_evals = {
                    "debug": debug_evals_official,
                    "generation": gen_evals_official,
                }
                candidate_codes = {
                    "debug": debug_codes,
                    "generation": gen_codes,
                }
                note = (
                    "oracle_routing_best1_each;"
                    + summarize_candidate_evals("debug", debug_evals_official, best_debug_idx)
                    + ";"
                    + summarize_candidate_evals("gen", gen_evals_official, best_gen_idx)
                    + f";selected_{selected_source}"
                )

            elif experiment.kind == "routing_verifier":
                vinfo = verifier_maps.get(experiment.verifier_source or "", {}).get(item["question_id"], {})
                tests = vinfo.get("verifier_tests", []) or []

                if not tests:
                    selected_source = "generation_sample_1"
                    selected_code = gen_codes[0] if gen_codes else ""
                    selected_eval = evaluate_code_on_tests(
                        selected_code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False
                    )
                    gen_eval = selected_eval
                    gen_code = selected_code
                    gen_raw = gen_raws[0] if gen_raws else ""
                    candidate_evals = [selected_eval]
                    candidate_codes = gen_codes
                    selected_sample_index = 0
                    note = "missing_or_empty_verifier_fallback_generation_sample_1"
                else:
                    debug_evals_official = [
                        evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False)
                        for code in debug_codes
                    ]
                    gen_evals_official = [
                        evaluate_code_on_tests(code, item["official_tests"], args.timeout, item["target_idx"], keep_details=False)
                        for code in gen_codes
                    ]

                    debug_evals_verifier = [
                        evaluate_code_on_tests(code, tests, args.timeout, 0, keep_details=False)
                        for code in debug_codes
                    ]
                    gen_evals_verifier = [
                        evaluate_code_on_tests(code, tests, args.timeout, 0, keep_details=False)
                        for code in gen_codes
                    ]

                    best_debug_idx = choose_best_candidate(debug_evals_verifier)
                    best_gen_idx = choose_best_candidate(gen_evals_verifier)

                    debug_v = debug_evals_verifier[best_debug_idx]
                    gen_v = gen_evals_verifier[best_gen_idx]

                    debug_eval = debug_evals_official[best_debug_idx]
                    gen_eval = gen_evals_official[best_gen_idx]
                    debug_code = debug_codes[best_debug_idx]
                    gen_code = gen_codes[best_gen_idx]
                    debug_raw = debug_raws[best_debug_idx]
                    gen_raw = gen_raws[best_gen_idx]

                    selected_branch = choose_candidate(debug_v, gen_v)
                    if selected_branch == "debug":
                        selected_source = f"debug_sample_{best_debug_idx + 1}"
                        selected_code = debug_code
                        selected_eval = debug_eval
                        selected_sample_index = best_debug_idx
                    else:
                        selected_source = f"generation_sample_{best_gen_idx + 1}"
                        selected_code = gen_code
                        selected_eval = gen_eval
                        selected_sample_index = best_gen_idx

                    candidate_evals = {
                        "debug_official": debug_evals_official,
                        "generation_official": gen_evals_official,
                        "debug_verifier": debug_evals_verifier,
                        "generation_verifier": gen_evals_verifier,
                    }
                    candidate_codes = {
                        "debug": debug_codes,
                        "generation": gen_codes,
                    }
                    note = (
                        "verifier_routing_best1_each;"
                        + summarize_candidate_evals("verifier_debug", debug_evals_verifier, best_debug_idx)
                        + ";"
                        + summarize_candidate_evals("verifier_gen", gen_evals_verifier, best_gen_idx)
                        + ";official_"
                        + summarize_candidate_evals("debug", debug_evals_official, best_debug_idx)
                        + ";official_"
                        + summarize_candidate_evals("gen", gen_evals_official, best_gen_idx)
                        + f";selected_{selected_source}"
                    )

            bto_source, bto_eval = compute_oracle_backtracking(item, selected_source, selected_eval)
            btr_source, btr_eval, rl_base_v, rl_sel_v = compute_rl_generated_backtracking(item, selected_source, selected_code, selected_eval, verifier_maps.get("rl_testgen", {}), args)
            write_result(item, {
                "selected_source": selected_source,
                "selected_code": selected_code,
                "nonbacktracking_eval": selected_eval,
                "backtracking_with_oracle_tests_source": bto_source,
                "backtracking_with_oracle_tests_eval": bto_eval,
                "backtracking_with_rl_generated_tests_source": btr_source,
                "backtracking_with_rl_generated_tests_eval": btr_eval,
                "rl_backtracking_baseline_verifier_eval": rl_base_v,
                "rl_backtracking_selected_verifier_eval": rl_sel_v,
                "debug_eval": json_safe_public_eval(debug_eval),
                "generation_eval": json_safe_public_eval(gen_eval),
                "debug_verifier_eval": debug_v,
                "generation_verifier_eval": gen_v,
                "debug_code": debug_code,
                "generation_code": gen_code,
                "debug_prompt_chars": debug_prompt_len,
                "generation_prompt_chars": gen_prompt_len,
                "debug_output_chars": len(debug_raw),
                "generation_output_chars": len(gen_raw),
                "candidate_evals": candidate_evals,
                "candidate_codes": candidate_codes,
                "selected_sample_index": selected_sample_index,
                "note": note,
            })
        pending = []

    try:
        for bug in tqdm(rows, desc=f"[{experiment.name}]", mininterval=args.tqdm_mininterval, disable=args.disable_tqdm):
            uid = str(bug.get("bug_uid", bug.get("task_id", "")))
            if uid in completed_ids:
                continue
            item = prepare_row(bug, problem_bank, args)
            if item is None:
                write_skip(uid, "missing_problem_or_tests")
                continue
            if item["bug_uid"] in completed_ids:
                continue
            has_failing = item["target_idx"] >= 0
            baseline_writer.writerow([
                item["bug_uid"], item["question_id"], item.get("bug_source_model_tag", ""),
                item["baseline_eval"]["full_suite_passed"], item["baseline_eval"]["full_suite_total"], f"{item['baseline_eval']['full_suite_acc_pct']:.2f}", int(bool(item["baseline_eval"].get("problem_passed"))),
                int(has_failing), item["target_idx"], "" if has_failing else "no_failing_official_test",
            ])
            baseline_fp.flush()
            if experiment.kind == "buggy":
                write_result(item, build_default_result(item, "baseline_only"))
                continue
            if not has_failing and experiment.kind in ("single_debug", "best2_debug_oracle", "best2_debug_verifier", "routing_oracle", "routing_verifier", "lcb_selfrepair"):
                write_result(item, build_default_result(item, "skipped_no_failing_official_test"))
                continue
            pending.append(item)
            if len(pending) >= args.batch_size:
                flush_pending()
        flush_pending()
    finally:
        attempts_fp.close(); baseline_fp.close()

    problems_seen = len(rows)
    final_non = build_final_metrics(experiment, "non_backtracking", problems_seen, evaluated, skipped, nb_passed, nb_total, nb_problem, nb_target, nb_counts, args)
    final_bto = build_final_metrics(experiment, "backtracking_with_oracle_tests", problems_seen, evaluated, skipped, bto_passed, bto_total, bto_problem, bto_target, bto_counts, args)
    final_btr = build_final_metrics(experiment, "backtracking_with_rl_generated_tests", problems_seen, evaluated, skipped, btr_passed, btr_total, btr_problem, btr_target, btr_counts, args)
    for p, obj in [(final_non_path, final_non), (final_bt_oracle_path, final_bto), (final_bt_rl_path, final_btr)]:
        with open(p, "w", encoding="utf-8") as fp:
            json.dump(obj, fp, ensure_ascii=False, indent=2, default=str)
    for out_dir in (out_bt_oracle, out_bt_rl):
        shutil.copy2(attempts_tsv, out_dir / attempts_tsv.name)
        shutil.copy2(baseline_tsv, out_dir / baseline_tsv.name)
        shutil.copy2(summary_jsonl, out_dir / summary_jsonl.name)
    final_all = {"experiment": experiment.name, "table_label": experiment.table_label, "non_backtracking": final_non, "backtracking_with_oracle_tests": final_bto, "backtracking_with_rl_generated_tests": final_btr}
    with open(final_all_path, "w", encoding="utf-8") as fp:
        json.dump(final_all, fp, ensure_ascii=False, indent=2, default=str)
    return final_all


def _bucket_from_baseline_pct(pct: float) -> str:
    if pct < 25.0:
        return "Easy"
    if pct < 50.0:
        return "Medium"
    if pct < 75.0:
        return "Hard"
    return "Very Hard"


def _empty_bucket_acc() -> Dict[str, Any]:
    return {
        "evaluated": 0,
        "test_passed": 0,
        "test_total": 0,
        "problem_passed": 0,
        "target_fixed": 0,
        "source_counts": {},
    }


def _add_bucket_record(acc: Dict[str, Any], eval_obj: Dict[str, Any], source: str, target_fixed: Optional[int] = None) -> None:
    acc["evaluated"] += 1
    acc["test_passed"] += int(eval_obj.get("full_suite_passed", 0) or 0)
    acc["test_total"] += int(eval_obj.get("full_suite_total", 0) or 0)
    acc["problem_passed"] += int(bool(eval_obj.get("problem_passed", False)))
    if target_fixed is None:
        target_fixed = int(bool(eval_obj.get("target_ok", False)))
    acc["target_fixed"] += int(bool(target_fixed))
    _add_count(acc["source_counts"], source or "unknown")


def write_bucketed_summary(log_dir: str, results: List[Dict[str, Any]]) -> None:
    """Write experiment x mode x baseline-pass-rate-bucket metrics.

    Buckets are assigned from the original buggy code's official LCB pass rate,
    i.e. baseline_eval.full_suite_acc_pct in summary_per_problem.jsonl.
    """
    mode_specs = [
        ("non_backtracking", "nonbacktracking_eval", "selected_source"),
        ("backtracking_with_oracle_tests", "backtracking_with_oracle_tests_eval", "backtracking_with_oracle_tests_source"),
        ("backtracking_with_rl_generated_tests", "backtracking_with_rl_generated_tests_eval", "backtracking_with_rl_generated_tests_source"),
    ]
    rows: List[Dict[str, Any]] = []
    bucket_json: Dict[str, Any] = {}
    for r in results:
        exp = r.get("experiment", "")
        table_label = r.get("table_label", exp)
        path = Path(log_dir) / exp / "non_backtracking" / "summary_per_problem.jsonl"
        accs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("event") != "final_metrics":
                    continue
                base = rec.get("baseline_eval", {}) or {}
                try:
                    baseline_pct = float(base.get("full_suite_acc_pct", 0.0) or 0.0)
                except Exception:
                    bp = int(base.get("full_suite_passed", 0) or 0)
                    bt = int(base.get("full_suite_total", 0) or 0)
                    baseline_pct = 100.0 * bp / max(1, bt)
                bucket = _bucket_from_baseline_pct(baseline_pct)
                for mode, eval_key, source_key in mode_specs:
                    ev = rec.get(eval_key, {}) or {}
                    key = (mode, bucket)
                    if key not in accs:
                        accs[key] = _empty_bucket_acc()
                    target_key = {
                        "non_backtracking": "nonbacktracking_target_fixed",
                        "backtracking_with_oracle_tests": "backtracking_with_oracle_tests_target_fixed",
                        "backtracking_with_rl_generated_tests": "backtracking_with_rl_generated_tests_target_fixed",
                    }[mode]
                    explicit_target_fixed = rec.get(target_key, None)
                    _add_bucket_record(
                        accs[key],
                        ev,
                        str(rec.get(source_key, rec.get("selected_source", "unknown"))),
                        explicit_target_fixed,
                    )
        bucket_json[exp] = {}
        for (mode, bucket), a in sorted(accs.items(), key=lambda x: (x[0][0], ["Easy","Medium","Hard","Very Hard"].index(x[0][1]))):
            evaluated = int(a["evaluated"])
            test_total = int(a["test_total"])
            source_counts = {k:int(v) for k,v in sorted(a["source_counts"].items())}
            row = {
                "experiment": exp,
                "table_label": table_label,
                "mode": mode,
                "bucket": bucket,
                "problem_pass_rate_pct": 100.0 * int(a["problem_passed"]) / max(1, evaluated),
                "problem_passed": int(a["problem_passed"]),
                "evaluated": evaluated,
                "overall_test_accuracy_pct": 100.0 * int(a["test_passed"]) / max(1, test_total),
                "test_passed": int(a["test_passed"]),
                "test_total": test_total,
                "targeted_fixed_count": int(a["target_fixed"]),
                "targeted_fixed_pct": 100.0 * int(a["target_fixed"]) / max(1, evaluated),
                "generation_count": int(source_counts.get("generation", 0)),
                "debug_count": int(source_counts.get("debug", 0)),
                "buggy_count": int(source_counts.get("buggy", 0)),
                "selfrepair_count": int(source_counts.get("selfrepair", 0)),
                "source_counts": source_counts,
            }
            rows.append(row)
            bucket_json.setdefault(exp, {}).setdefault(mode, {})[bucket] = row
    out_json = Path(log_dir) / "all_selected_experiments_summary_by_bucket.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(bucket_json, fp, ensure_ascii=False, indent=2, default=str)
    out_tsv = Path(log_dir) / "all_selected_experiments_summary_by_bucket.tsv"
    with open(out_tsv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerow([
            "experiment", "table_label", "mode", "bucket", "problem_pass_rate_pct", "problem_passed", "evaluated",
            "overall_test_accuracy_pct", "test_passed", "test_total", "targeted_fixed_count", "targeted_fixed_pct",
            "generation_count", "debug_count", "buggy_count", "selfrepair_count", "source_counts",
        ])
        for row in rows:
            w.writerow([
                row["experiment"], row["table_label"], row["mode"], row["bucket"], f"{row['problem_pass_rate_pct']:.4f}", row["problem_passed"], row["evaluated"],
                f"{row['overall_test_accuracy_pct']:.4f}", row["test_passed"], row["test_total"], row["targeted_fixed_count"], f"{row['targeted_fixed_pct']:.4f}",
                row["generation_count"], row["debug_count"], row["buggy_count"], row["selfrepair_count"], json.dumps(row["source_counts"], sort_keys=True),
            ])


def write_combined_summary(log_dir: str, results: List[Dict[str, Any]]) -> None:
    ensure_dir(log_dir)
    with open(Path(log_dir) / "all_selected_experiments_summary.json", "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2, default=str)
    tsv = Path(log_dir) / "all_selected_experiments_summary.tsv"
    with open(tsv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerow([
            "experiment", "table_label", "mode", "problem_pass_rate_pct", "problem_passed", "evaluated",
            "overall_test_accuracy_pct", "test_passed", "test_total", "targeted_fixed_count", "targeted_fixed_pct",
            "generation_count", "debug_count", "buggy_count", "selfrepair_count", "source_counts",
        ])
        for r in results:
            for mode_key in ("non_backtracking", "backtracking_with_oracle_tests", "backtracking_with_rl_generated_tests"):
                if mode_key not in r:
                    continue
                m = r[mode_key]
                sc = m.get("source_counts", {}) or {}
                w.writerow([
                    r["experiment"], r["table_label"], m["mode"], f"{m['problem_pass_rate_pct']:.4f}", m["problem_passed_count"], m["evaluated"],
                    f"{m['overall_test_accuracy_pct']:.4f}", m["final_full_suite_passed_total"], m["final_full_suite_total_tests"],
                    m.get("targeted_fixed_count", 0), f"{m.get('targeted_fixed_pct', 0.0):.4f}",
                    m.get("generation_count", 0), m.get("debug_count", 0), m.get("buggy_count", 0), int(sc.get("selfrepair", 0)), json.dumps(sc, sort_keys=True),
                ])
    write_bucketed_summary(log_dir, results)

# -------------------------
# CLI
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--experiments", nargs="*", default=[], choices=sorted(EXPERIMENTS.keys()))
    ap.add_argument("--release_version", type=str, default=DEFAULT_RELEASE_VERSION)
    ap.add_argument("--codegen_root", type=str, default=DEFAULT_CODEGEN_ROOT)
    ap.add_argument("--limit_tasks", type=int, default=0, help="Limit combined buggy rows, not unique LCB problems.")
    ap.add_argument("--num_shards", type=int, default=1, help="Number of dataset shards for parallel runs.")
    ap.add_argument("--shard_index", type=int, default=0, help="This process shard index in [0, num_shards).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target_policy", choices=["first", "random"], default="first")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL_ID)
    ap.add_argument("--rl_peft_checkpoint", type=str, default=DEFAULT_RL_PEFT_CHECKPOINT)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--deterministic_test", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--prompt_max_tokens", type=int, default=32768)
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
    args.codegen_root = expand_path(args.codegen_root)
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
    experiments = [EXPERIMENTS[n] for n in exp_names]

    print(f"[lcb] Loading problem bank release={args.release_version}", flush=True)
    problem_bank = load_lcb_problem_bank(args.release_version)
    print(f"[lcb] problems loaded={len(problem_bank)}", flush=True)

    print(f"[bugs] Loading and combining bug farms from {args.codegen_root}", flush=True)
    bug_rows, combined_path = load_bug_farms(args.codegen_root)
    bug_rows = [r for r in bug_rows if str(r.get("question_id", "")) in problem_bank]
    if args.limit_tasks and args.limit_tasks > 0:
        bug_rows = bug_rows[: args.limit_tasks]

    if int(args.num_shards) < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError(f"--shard_index must be in [0, num_shards), got {args.shard_index}/{args.num_shards}")

    if int(args.num_shards) > 1:
        before_shard = len(bug_rows)
        bug_rows = [r for i, r in enumerate(bug_rows) if (i % int(args.num_shards)) == int(args.shard_index)]
        print(
            f"[shard] shard_index={args.shard_index} num_shards={args.num_shards} "
            f"rows={len(bug_rows)}/{before_shard}",
            flush=True,
        )

    print(f"[bugs] combined_bug_rows={len(bug_rows)} combined_file={combined_path}", flush=True)

    need_base, need_rl = needed_models(experiments)
    need_base_verifier = any(e.verifier_source == "base_testgen" for e in experiments)
    need_rl_verifier = True  # always needed for generated-test backtracking.

    verifier_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if need_base_verifier:
        print(f"[verifier] Loading base tests: {args.base_testgen_jsonl}", flush=True)
        verifier_maps["base_testgen"] = load_verifier_tests_map(args.base_testgen_jsonl)
        print(f"[verifier] base qids={len(verifier_maps['base_testgen'])}", flush=True)
    if need_rl_verifier:
        print(f"[verifier] Loading RL tests: {args.rl_testgen_jsonl}", flush=True)
        verifier_maps["rl_testgen"] = load_verifier_tests_map(args.rl_testgen_jsonl)
        print(f"[verifier] rl qids={len(verifier_maps['rl_testgen'])}", flush=True)

    with open(Path(args.log_dir) / "run_args.json", "w", encoding="utf-8") as fp:
        json.dump(vars(args) | {"combined_bug_farm": combined_path}, fp, ensure_ascii=False, indent=2, default=str)

    runners: Dict[str, ModelRunner] = {}
    try:
        if need_base and need_rl:
            if not args.rl_peft_checkpoint or not os.path.isdir(args.rl_peft_checkpoint):
                raise FileNotFoundError(f"RL PEFT checkpoint not found: {args.rl_peft_checkpoint}")
            print(f"[model] Loading shared Qwen PEFT once: base={args.base_model_id} adapter={args.rl_peft_checkpoint}", flush=True)
            shared = load_model_and_tokenizer(
                args.base_model_id,
                peft_checkpoint=args.rl_peft_checkpoint,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                prompt_max_tokens=args.prompt_max_tokens,
            )
            runners["rl"] = shared
            runners["base"] = ModelRunner(
                shared.model,
                shared.tokenizer,
                model_id=args.base_model_id,
                peft_checkpoint=None,
                prompt_max_tokens=args.prompt_max_tokens,
                disable_adapter=True,
            )
            print("[model] Shared model ready: runners['base'] uses disable_adapter(); runners['rl'] uses adapter", flush=True)
        else:
            if need_base:
                print(f"[model] Loading base model: {args.base_model_id}", flush=True)
                runners["base"] = load_model_and_tokenizer(args.base_model_id, peft_checkpoint=None, dtype=args.dtype, trust_remote_code=args.trust_remote_code, prompt_max_tokens=args.prompt_max_tokens)
            if need_rl:
                if not args.rl_peft_checkpoint or not os.path.isdir(args.rl_peft_checkpoint):
                    raise FileNotFoundError(f"RL PEFT checkpoint not found: {args.rl_peft_checkpoint}")
                print(f"[model] Loading RL PEFT: base={args.base_model_id} adapter={args.rl_peft_checkpoint}", flush=True)
                runners["rl"] = load_model_and_tokenizer(args.base_model_id, peft_checkpoint=args.rl_peft_checkpoint, dtype=args.dtype, trust_remote_code=args.trust_remote_code, prompt_max_tokens=args.prompt_max_tokens)

        all_results: List[Dict[str, Any]] = []
        for exp in experiments:
            print(f"\n===== Running {exp.name}: {exp.table_label} =====", flush=True)
            res = run_experiment(exp, args, bug_rows, problem_bank, runners, verifier_maps)
            all_results.append(res)
            write_combined_summary(args.log_dir, all_results)
        print("\n=== LCB table experiments complete ===", flush=True)
        print(f"Summary: {Path(args.log_dir) / 'all_selected_experiments_summary.tsv'}", flush=True)
    finally:
        seen_model_ids = set()
        for r in runners.values():
            mid = id(getattr(r, "model", None))
            if mid in seen_model_ids:
                continue
            seen_model_ids.add(mid)
            free_runner(r)


if __name__ == "__main__":
    main()

