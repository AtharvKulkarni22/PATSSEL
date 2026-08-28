#!/usr/bin/env python3
"""
Unified LeetCode table runner.

This file combines the main experiment families used to build the generation/debug
comparison tables:

  * Buggy Code Acc baseline
  * Base/RL Debug Only
  * Base/RL Gen Only
  * Base Gen Best-of-2 Oracle Selection
  * RL Debug Best-of-2 Oracle Selection
  * Base/RL/RL+Base Oracle Routing
  * Base/RL/RL+Base Verifier Routing with either base-testgen or RL-testgen verifier tests

The LeetCode evaluator is provided in patssel/evaluation/leetcode.py.

Examples
--------
Smoke test, base debug only:
  python leetcode_table_unified_runner.py \
    --experiments base_debug_only \
    --log_dir /path/to/results \
    --base_model_id Qwen/Qwen3-8B \
    --limit_per_subset 5

Run all table experiments:
  python leetcode_table_unified_runner.py \
    --all \
    --log_dir /path/to/results \
    --base_model_id Qwen/Qwen3-8B \
    --rl_peft_checkpoint /path/to/checkpoint-3000 \
    --base_testgen_verifier_jsonl /path/to/base_testgen/per_problem_detailed.jsonl \
    --rl_testgen_verifier_jsonl /path/to/rl_testgen/per_problem_detailed.jsonl
"""

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import csv
import json
import random
import shutil
import argparse
import hashlib
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from patssel.evaluation.leetcode import (
    eval_all_asserts_with_output,
    eval_single_assert_with_output,
    extract_assert_texts,
    resolve_entry_point,
)

# -------------------------
# Defaults matching old code
# -------------------------
DEFAULT_SEED = 1012
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20

DEFAULT_SUBSET_TSV_EASY = str(_REPO_ROOT / "data/leetcode/test/q1_0_25.tsv")
DEFAULT_SUBSET_TSV_MEDIUM = str(_REPO_ROOT / "data/leetcode/test/q2_25_50.tsv")
DEFAULT_SUBSET_TSV_HARD = str(_REPO_ROOT / "data/leetcode/test/q3_50_75.tsv")
DEFAULT_SUBSET_TSV_EXTRA_HARD = str(_REPO_ROOT / "data/leetcode/test/q4_75_99.tsv")

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_RL_PEFT_CHECKPOINT = str(_REPO_ROOT / "checkpoints/debugger")
DEFAULT_LEETCODE_DATASET_NAME = "newfacade/LeetCodeDataset"
DEFAULT_LEETCODE_SPLIT = "train"
DEFAULT_BASE_TESTGEN_VERIFIER_JSONL = str(_REPO_ROOT / "outputs/verifier_tests/leetcode/base/per_problem_detailed.jsonl")
DEFAULT_RL_TESTGEN_VERIFIER_JSONL = str(_REPO_ROOT / "outputs/verifier_tests/leetcode/rl/per_problem_detailed.jsonl")

FENCE_PY_RE = re.compile(r"```python\s*([\s\S]*?)\s*```", re.IGNORECASE)
GENERIC_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+\-]*\s*\n([\s\S]*?)\n```$", re.S)


# -------------------------
# General utilities
# -------------------------
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_int_from_str(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def split_df_half_deterministic(df: pd.DataFrame, seed: int, split_name: str, mode: str) -> pd.DataFrame:
    if mode not in ("dev", "test", "all"):
        raise ValueError("split mode must be one of: dev, test, all")
    if mode == "all":
        return df.reset_index(drop=True)
    rs = np.random.RandomState(seed + stable_int_from_str(split_name))
    idx = np.arange(len(df))
    rs.shuffle(idx)
    mid = len(idx) // 2
    chosen = idx[mid:] if mode == "test" else idx[:mid]
    return df.iloc[chosen].reset_index(drop=True)


def save_jsonl_line(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_subset_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "prediction" in df.columns and "buggy_code" not in df.columns:
        df = df.rename(columns={"prediction": "buggy_code"})
    required = [
        "task_id", "question_id", "buggy_code", "test_code",
        "selected_assert_index", "selected_assert_text",
        "baseline_total", "baseline_failed", "baseline_passed",
        "entry_point", "query",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Subset TSV missing required columns: {missing}. Present: {list(df.columns)}")
    for c in ["selected_assert_index", "baseline_total", "baseline_failed", "baseline_passed", "pass", "fail", "total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df.fillna("")


def extract_fenced_code(text: str) -> str:
    if not text:
        return ""
    matches = FENCE_PY_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return ""


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
        # Qwen chat templates support enable_thinking=False. Fall back cleanly
        # for tokenizer versions that do not expose that template argument.
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

    @torch.no_grad()
    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        deterministic: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Tuple[str, int]]:
        if not batch_messages:
            return []
        prompt_texts = [self.render_chat_prompt(m) for m in batch_messages]
        prompt_lens = [len(json.dumps(m, ensure_ascii=False)) for m in batch_messages]
        enc = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=False)
        dev = first_shard_device(self.model)
        enc = {k: v.to(dev) for k, v in enc.items()}
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if deterministic:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        outputs = self.model.generate(**enc, **gen_kwargs)
        results: List[Tuple[str, int]] = []
        for i in range(outputs.shape[0]):
            gen_ids = outputs[i][input_lengths[i]:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append((gen_text, prompt_lens[i]))
        return results


def load_backbone_model(model_id: str, dtype: str = "auto") -> AutoModelForCausalLM:
    torch_dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True)
    model.eval()
    return model


def load_model_and_tokenizer(model_id: str, peft_checkpoint: Optional[str] = None, dtype: str = "auto") -> ModelRunner:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if peft_checkpoint:
        if PeftModel is None:
            raise RuntimeError("peft is not available, but a PEFT checkpoint was requested.")
        base = load_backbone_model(model_id, dtype=dtype)
        model = PeftModel.from_pretrained(base, peft_checkpoint, torch_dtype=("auto" if dtype == "auto" else getattr(torch, dtype)))
        model.eval()
    else:
        model = load_backbone_model(model_id, dtype=dtype)
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
# Prompt builders
# -------------------------
def build_debug_messages(buggy_or_current_code: str, assert_text: str) -> List[Dict[str, str]]:
    system_msg = (
        "You are a Python bug-fixing assistant.\n\n"
        "You MUST respond with the exact structure below. Every section is REQUIRED.\n"
        "Write naturally and with as much detail as needed (no length limit).\n\n"
        "1) CODE INTENT\n"
        "2) BUG REVIEW\n"
        "3) FIX PLAN\n"
        "4) FINAL FIX\n"
        "   Output ONLY the fixed function/class in exactly ONE Python fenced code block.\n"
        "   The code block MUST be the LAST thing in the response.\n"
        "   Keep the same function/class name and signature.\n"
        "   Do NOT change imports.\n"
    )
    user_msg = (
        "Fix the buggy function/class so that the failing unit test passes, also try to fix any other issues in the code.\n"
        "- Keep the same function/class name and signature.\n"
        "- Do NOT change imports.\n\n"
        "Buggy function/class:\n"
        f"{textwrap.dedent(buggy_or_current_code).strip()}\n\n"
        "Failing unit test:\n"
        f"{assert_text}\n"
    )
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]


def build_generation_messages(query: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": str(query or "")}]


# -------------------------
# Dataset and verifier loading
# -------------------------
def load_leetcode_query_map(dataset_name: str, split: str) -> Dict[Any, Dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    if "task_id" not in ds.column_names:
        raise ValueError(f"{dataset_name} split={split} is missing task_id. Columns={ds.column_names}")
    return {row["task_id"]: row for row in ds}


def get_generation_query(task_id: Any, ds_map: Dict[Any, Dict[str, Any]]) -> str:
    row = ds_map.get(task_id)
    if row is None:
        row = ds_map.get(str(task_id))
    if row is None:
        try:
            row = ds_map.get(int(task_id))
        except Exception:
            row = None
    if row is None:
        return ""
    return str(row.get("query", "") or row.get("prompt", "") or "")


def make_verifier_key(task_id: Any, question_id: Any, source_tsv: str) -> str:
    return f"{str(task_id)}|||{str(question_id)}|||{str(source_tsv)}"


def load_verifier_tests_map(verifier_jsonl: str) -> Dict[str, Dict[str, Any]]:
    """Load generated verifier tests.

    Historical verifier records may store a different absolute source_tsv path, so we index each record
    using several aliases: full source_tsv, basename(source_tsv), and task/qid only.
    This prevents path migration from breaking verifier lookup.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not verifier_jsonl:
        raise ValueError("verifier_jsonl is empty")
    with open(verifier_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            task_id = rec.get("task_id", "")
            question_id = rec.get("question_id", "")
            source_tsv = str(rec.get("source_tsv", "") or "")
            clean = str(rec.get("generated_test_code_eval", "") or rec.get("generated_test_code_clean", "") or "")
            raw = str(rec.get("generated_test_code_raw_extracted", "") or "")
            gc = rec.get("gold_correctness", {}) or {}
            try:
                num_asserts = int(gc.get("num_asserts", 0))
            except Exception:
                num_asserts = 0
            info = {
                "verifier_test_code": clean if clean.strip() else raw,
                "generated_wellformed": bool(rec.get("generated_wellformed", False)),
                "num_asserts": num_asserts,
                "record": rec,
            }
            keys = [
                make_verifier_key(task_id, question_id, source_tsv),
                make_verifier_key(task_id, question_id, os.path.basename(source_tsv)),
                _row_key(task_id, question_id),
            ]
            for key in keys:
                if key not in out:
                    out[key] = info
    return out


def get_verifier_info(verifier_map: Dict[str, Dict[str, Any]], task_id: Any, question_id: Any, subset_tsv: str) -> Optional[Dict[str, Any]]:
    """Path-robust verifier lookup after migrating subset TSV paths."""
    for key in (
        make_verifier_key(task_id, question_id, subset_tsv),
        make_verifier_key(task_id, question_id, os.path.basename(str(subset_tsv))),
        _row_key(task_id, question_id),
    ):
        v = verifier_map.get(key)
        if v is not None:
            return v
    return None


# -------------------------
# Evaluation helpers
# -------------------------
def evaluate_candidate(query_text: str, candidate_code: str, test_code: str, entry_point: Optional[str], selected_assert_index: int) -> Dict[str, Any]:
    try:
        sel = eval_single_assert_with_output(query_text, candidate_code, test_code, entry_point, selected_assert_index)
    except Exception:
        sel = {"ok": False, "stdout": "", "error_type": "eval_exception", "error_msg": ""}
    try:
        full = eval_all_asserts_with_output(query_text, candidate_code, test_code, entry_point)
    except Exception as e:
        full = {"results": [], "total": 0, "error": f"{type(e).__name__}: {e}"}
    results = full.get("results", [])
    total = int(full.get("total", len(results)))
    passed = int(sum(1 for r in results if r.get("ok")))
    return {
        "target_ok": bool(sel.get("ok", False)),
        "stdout": sel.get("stdout", ""),
        "error_type": sel.get("error_type", ""),
        "error_msg": sel.get("error_msg", ""),
        "full_suite_passed": passed,
        "full_suite_total": total,
        "full_suite_acc_pct": (100.0 * passed / max(1, total)) if total > 0 else 0.0,
    }


def choose_candidate(debug_eval: Dict[str, Any], gen_eval: Dict[str, Any]) -> str:
    if gen_eval["full_suite_passed"] > debug_eval["full_suite_passed"]:
        return "generation"
    if gen_eval["full_suite_passed"] < debug_eval["full_suite_passed"]:
        return "debug"
    if gen_eval["target_ok"] and not debug_eval["target_ok"]:
        return "generation"
    return "debug"


def choose_best_oracle_candidate(candidate_evals: List[Dict[str, Any]]) -> int:
    """Return the index of the candidate with the highest oracle-test pass rate.

    Ties are broken by the selected failing assert, then by earliest sample.
    """
    if not candidate_evals:
        raise ValueError("candidate_evals must be non-empty")
    best_idx = 0
    best_key = (
        float(candidate_evals[0].get("full_suite_acc_pct", 0.0)),
        int(bool(candidate_evals[0].get("target_ok", False))),
    )
    for i, ev in enumerate(candidate_evals[1:], start=1):
        key = (
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


def baseline_for_row(row: pd.Series) -> Dict[str, Any]:
    task_id = row["task_id"]
    question_id = row["question_id"]
    current_code = str(row["buggy_code"] or "")
    test_code = str(row["test_code"] or "")
    entry_point = row["entry_point"] if pd.notna(row["entry_point"]) and row["entry_point"] != "" else None
    query_text = row.get("query", "")
    aidx = int(row["selected_assert_index"])
    assert_text = str(row["selected_assert_text"] or "")
    note = ""

    try:
        full_base = eval_all_asserts_with_output(query_text, current_code, test_code, entry_point)
        base_results = full_base.get("results", [])
    except Exception:
        base_results = []
        full_base = {"results": [], "total": 0}

    baseline_total = int(full_base.get("total", len(base_results)))
    baseline_passed = int(sum(1 for r in base_results if r.get("ok")))
    baseline_acc_pct = (100.0 * baseline_passed / max(1, baseline_total)) if baseline_total > 0 else 0.0

    baseline_selected_assert_failing = 0
    if baseline_total > 0 and 0 <= aidx < baseline_total:
        try:
            res_sel = eval_single_assert_with_output(query_text, current_code, test_code, entry_point, aidx)
            baseline_selected_assert_failing = 0 if bool(res_sel.get("ok", False)) else 1
        except Exception:
            baseline_selected_assert_failing = 1

    if baseline_selected_assert_failing == 0:
        failing_indices = sorted([r["index"] for r in base_results if not r.get("ok")])
        if failing_indices:
            old_idx = aidx
            aidx = int(failing_indices[0])
            assert_texts = extract_assert_texts(test_code)
            assert_text = assert_texts[aidx] if aidx < len(assert_texts) else f"assert #{aidx}"
            note = f"reselected_assert_from_{old_idx}_to_{aidx}"
            baseline_selected_assert_failing = 1
        else:
            note = "skipped_no_failing_asserts"

    return {
        "task_id": task_id,
        "question_id": question_id,
        "aidx": aidx,
        "assert_text": assert_text,
        "current_code": current_code,
        "test_code": test_code,
        "entry_point": entry_point,
        "query_text": query_text,
        "baseline_selected_assert_failing": baseline_selected_assert_failing,
        "baseline_passed": baseline_passed,
        "baseline_total": baseline_total,
        "baseline_acc_pct": baseline_acc_pct,
        "note": note,
    }


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


# -------------------------
# Aggregation helpers
# -------------------------
def build_final_dict(
    experiment: Experiment,
    mode: str,
    subset_tsv: str,
    split_mode: str,
    problems_count: int,
    evaluated_count: int,
    skipped_count: int,
    full_passed_total: int,
    full_total_asserts: int,
    targeted_fixed_count: int,
    source_counts: Dict[str, int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    acc = (100.0 * full_passed_total / max(1, full_total_asserts)) if full_total_asserts > 0 else 0.0
    targeted_pct = (100.0 * targeted_fixed_count / max(1, problems_count)) if problems_count > 0 else 0.0
    return {
        "experiment": experiment.name,
        "table_label": experiment.table_label,
        "kind": experiment.kind,
        "subset_tsv": subset_tsv,
        "split_mode": split_mode,
        "mode": mode,
        "problems": int(problems_count),
        "evaluated": int(evaluated_count),
        "skipped_count": int(skipped_count),
        "final_full_suite_passed_total": int(full_passed_total),
        "final_full_suite_total_asserts": int(full_total_asserts),
        "overall_accuracy_pct": float(acc),
        "targeted_fixed_count": int(targeted_fixed_count),
        "targeted_fixed_pct": float(targeted_pct),
        "source_counts": {k: int(v) for k, v in sorted(source_counts.items())},
        "batch_size": int(args.batch_size),
        "deterministic_generation": bool(args.deterministic_test),
        "seed": int(args.seed),
        "gen_params": {
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
        },
    }


def sum_final_metrics(experiment: Experiment, per_subset: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    agg = {
        "problems": 0,
        "evaluated": 0,
        "skipped_count": 0,
        "final_full_suite_passed_total": 0,
        "final_full_suite_total_asserts": 0,
        "targeted_fixed_count": 0,
        "source_counts": {},
    }
    for m in per_subset:
        agg["problems"] += int(m.get("problems", 0))
        agg["evaluated"] += int(m.get("evaluated", 0))
        agg["skipped_count"] += int(m.get("skipped_count", 0))
        agg["final_full_suite_passed_total"] += int(m.get("final_full_suite_passed_total", 0))
        agg["final_full_suite_total_asserts"] += int(m.get("final_full_suite_total_asserts", 0))
        agg["targeted_fixed_count"] += int(m.get("targeted_fixed_count", 0))
        for k, v in (m.get("source_counts", {}) or {}).items():
            agg["source_counts"][k] = agg["source_counts"].get(k, 0) + int(v)
    total = agg["final_full_suite_total_asserts"]
    passed = agg["final_full_suite_passed_total"]
    return {
        "experiment": experiment.name,
        "table_label": experiment.table_label,
        "kind": experiment.kind,
        "mode": mode,
        "problems": int(agg["problems"]),
        "evaluated": int(agg["evaluated"]),
        "skipped_count": int(agg["skipped_count"]),
        "final_full_suite_passed_total": int(passed),
        "final_full_suite_total_asserts": int(total),
        "overall_accuracy_pct": float((100.0 * passed / max(1, total)) if total > 0 else 0.0),
        "targeted_fixed_count": int(agg["targeted_fixed_count"]),
        "targeted_fixed_pct": float((100.0 * agg["targeted_fixed_count"] / max(1, agg["problems"])) if agg["problems"] else 0.0),
        "source_counts": {k: int(v) for k, v in sorted(agg["source_counts"].items())},
    }


# -------------------------
# Core subset runners
# -------------------------
def load_subset(args: argparse.Namespace, subset_tsv: str) -> pd.DataFrame:
    problems = pd.read_csv(subset_tsv, sep="\t", dtype=str)
    problems = _normalize_subset_columns(problems)
    for c in ["selected_assert_index", "baseline_total", "baseline_failed", "baseline_passed"]:
        if c in problems.columns:
            problems[c] = problems[c].astype(int)
    problems = split_df_half_deterministic(problems, seed=args.seed, split_name=os.path.basename(subset_tsv), mode=args.split_mode)
    if args.limit_per_subset and len(problems) > args.limit_per_subset:
        problems = problems.iloc[:args.limit_per_subset].reset_index(drop=True)

    start_idx = max(0, int(getattr(args, "start_idx", 0) or 0))
    end_idx = int(getattr(args, "end_idx", 0) or 0)
    if start_idx or end_idx:
        if end_idx <= 0 or end_idx > len(problems):
            end_idx = len(problems)
        problems = problems.iloc[start_idx:end_idx].reset_index(drop=True)

    return problems


def _row_key(task_id: Any, question_id: Any) -> str:
    return f"{str(task_id)}|||{str(question_id)}"


def _log_has_content(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _open_tsv_for_resume(path: str, header: List[str], force: bool) -> Tuple[Any, csv.writer]:
    if force and os.path.exists(path):
        os.remove(path)
    ensure_dir(os.path.dirname(path))
    need_header = not _log_has_content(path)
    fp = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fp, delimiter="\t")
    if need_header:
        writer.writerow(header)
        fp.flush()
    return fp, writer


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


def _accumulate_eval(state: Dict[str, Any], prefix: str, eval_dict: Dict[str, Any], source: str) -> None:
    state[f"{prefix}_passed"] += _metric_int(eval_dict, "full_suite_passed")
    state[f"{prefix}_total"] += _metric_int(eval_dict, "full_suite_total")
    state[f"{prefix}_target"] += int(_metric_bool(eval_dict, "target_ok"))
    _add_count(state[f"{prefix}_counts"], source)


def load_resume_state(summary_jsonl: str, experiment_name: str) -> Dict[str, Any]:
    """Load row-level resume state from summary_per_problem.jsonl."""
    empty = {
        "completed_ids": set(),
        "evaluated": 0,
        "skipped": 0,
        "nb_passed": 0, "nb_total": 0, "nb_target": 0, "nb_counts": {},
        "bt_oracle_passed": 0, "bt_oracle_total": 0, "bt_oracle_target": 0, "bt_oracle_counts": {},
        "bt_rl_passed": 0, "bt_rl_total": 0, "bt_rl_target": 0, "bt_rl_counts": {},
    }
    if not os.path.exists(summary_jsonl):
        return empty
    by_task: Dict[str, Dict[str, Any]] = {}
    with open(summary_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("experiment") not in (None, "", experiment_name):
                continue
            tid = str(rec.get("task_id", "")).strip()
            qid = str(rec.get("question_id", "")).strip()
            if not tid:
                continue
            key = _row_key(tid, qid)
            if rec.get("event") in ("final_metrics", "skipped_prepare_row"):
                by_task[key] = rec
    state = empty
    state["completed_ids"] = set(by_task.keys())
    for rec in by_task.values():
        if rec.get("event") == "skipped_prepare_row":
            state["skipped"] += 1
            continue
        if rec.get("event") != "final_metrics":
            continue
        state["evaluated"] += 1
        nb_eval = rec.get("nonbacktracking_eval", {}) or {}
        bt_oracle_eval = rec.get("backtracking_with_oracle_tests_eval", {}) or rec.get("backtracking_eval", {}) or {}
        bt_rl_eval = rec.get("backtracking_with_rl_generated_tests_eval", {}) or {}
        _accumulate_eval(state, "nb", nb_eval, str(rec.get("selected_source", "unknown")))
        _accumulate_eval(state, "bt_oracle", bt_oracle_eval, str(rec.get("backtracking_with_oracle_tests_source", rec.get("backtracking_source", "unknown"))))
        _accumulate_eval(state, "bt_rl", bt_rl_eval, str(rec.get("backtracking_with_rl_generated_tests_source", "unknown")))
    return state


def open_writers(out_dir: str, experiment: Experiment, force: bool) -> Tuple[Any, Any, csv.writer, csv.writer, str, str, str, str]:
    ensure_dir(out_dir)
    attempts_tsv = os.path.join(out_dir, "attempts_log_with_full_code_and_suite.tsv")
    baseline_tsv = os.path.join(out_dir, "baseline_metrics.tsv")
    summary_jsonl = os.path.join(out_dir, "summary_per_problem.jsonl")
    final_metrics_json = os.path.join(out_dir, "final_metrics.json")
    if force:
        for path in (attempts_tsv, baseline_tsv, summary_jsonl, final_metrics_json):
            if os.path.exists(path):
                os.remove(path)
    attempts_header = [
        "experiment", "task_id", "question_id", "assert_index",
        "baseline_full_suite_passed", "baseline_full_suite_total", "baseline_full_suite_acc_pct",
        "debug_full_suite_passed", "debug_full_suite_total", "debug_full_suite_acc_pct",
        "gen_full_suite_passed", "gen_full_suite_total", "gen_full_suite_acc_pct",
        "verifier_debug_passed", "verifier_debug_total", "verifier_gen_passed", "verifier_gen_total",
        "rl_backtracking_baseline_verifier_passed", "rl_backtracking_baseline_verifier_total",
        "rl_backtracking_selected_verifier_passed", "rl_backtracking_selected_verifier_total",
        "selected_source", "nonbacktracking_passed", "nonbacktracking_total", "nonbacktracking_acc_pct",
        "backtracking_with_oracle_tests_source", "backtracking_with_oracle_tests_passed", "backtracking_with_oracle_tests_total", "backtracking_with_oracle_tests_acc_pct",
        "backtracking_with_rl_generated_tests_source", "backtracking_with_rl_generated_tests_passed", "backtracking_with_rl_generated_tests_total", "backtracking_with_rl_generated_tests_acc_pct",
        "target_ok", "debug_prompt_chars", "gen_prompt_chars", "debug_output_chars", "gen_output_chars",
        "entry_point", "note", "debug_code_full", "generation_code_full", "selected_code_full",
    ]
    baseline_header = [
        "task_id", "question_id", "selected_assert_index", "baseline_selected_assert_failing",
        "baseline_full_suite_passed", "baseline_full_suite_total", "baseline_full_suite_acc_pct", "note",
    ]
    attempts_fp, attempts_writer = _open_tsv_for_resume(attempts_tsv, attempts_header, force=False)
    baseline_fp, baseline_writer = _open_tsv_for_resume(baseline_tsv, baseline_header, force=False)
    return attempts_fp, baseline_fp, attempts_writer, baseline_writer, attempts_tsv, baseline_tsv, summary_jsonl, final_metrics_json


def _baseline_eval_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "target_ok": False,
        "full_suite_passed": item["baseline_passed"],
        "full_suite_total": item["baseline_total"],
        "full_suite_acc_pct": item["baseline_acc_pct"],
    }


def _find_verifier_aidx(query_text: str, candidate_code: str, verifier_test_code: str, entry_point: Optional[str]) -> int:
    try:
        base_verifier = eval_all_asserts_with_output(query_text, candidate_code, verifier_test_code, entry_point)
        base_v_results = base_verifier.get("results", [])
        failing_v = sorted([r["index"] for r in base_v_results if not r.get("ok")])
        return int(failing_v[0]) if failing_v else 0
    except Exception:
        return 0


def _compute_rl_generated_backtracking(
    item: Dict[str, Any],
    subset_tsv: str,
    selected_source: str,
    selected_code: str,
    selected_eval: Dict[str, Any],
    baseline_eval: Dict[str, Any],
    rl_backtracking_verifier_map: Optional[Dict[str, Dict[str, Any]]],
    note: str,
) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    if selected_source == "buggy":
        return "buggy", baseline_eval, None, None, note
    if not rl_backtracking_verifier_map:
        note = (note + ";" if note else "") + "rl_generated_backtracking_missing_verifier_map"
        return "buggy", baseline_eval, None, None, note
    vkey = make_verifier_key(item["task_id"], item["question_id"], subset_tsv)
    vinfo = get_verifier_info(rl_backtracking_verifier_map, item["task_id"], item["question_id"], subset_tsv)
    if vinfo is None or not str(vinfo.get("verifier_test_code", "")).strip():
        note = (note + ";" if note else "") + "rl_generated_backtracking_missing_or_empty_verifier"
        return "buggy", baseline_eval, None, None, note
    verifier_test_code = str(vinfo["verifier_test_code"])
    verifier_aidx = _find_verifier_aidx(item["query_text"], item["current_code"], verifier_test_code, item["entry_point"])
    baseline_v_eval = evaluate_candidate(item["query_text"], item["current_code"], verifier_test_code, item["entry_point"], verifier_aidx)
    selected_v_eval = evaluate_candidate(item["query_text"], selected_code, verifier_test_code, item["entry_point"], verifier_aidx)
    if selected_v_eval["full_suite_passed"] > baseline_v_eval["full_suite_passed"]:
        return selected_source, selected_eval, baseline_v_eval, selected_v_eval, note
    return "buggy", baseline_eval, baseline_v_eval, selected_v_eval, note


def run_subset(
    experiment: Experiment,
    subset_tsv: str,
    subset_label: str,
    out_dir_non_bt: str,
    out_dir_bt_oracle: str,
    out_dir_bt_rl: str,
    args: argparse.Namespace,
    leetcode_map: Optional[Dict[Any, Dict[str, Any]]],
    debug_runner: Optional[ModelRunner],
    generation_runner: Optional[ModelRunner],
    verifier_map: Optional[Dict[str, Dict[str, Any]]],
    verifier_jsonl: Optional[str],
    rl_backtracking_verifier_map: Optional[Dict[str, Dict[str, Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    final_non_path = os.path.join(out_dir_non_bt, "final_metrics.json")
    final_bt_oracle_path = os.path.join(out_dir_bt_oracle, "final_metrics.json")
    final_bt_rl_path = os.path.join(out_dir_bt_rl, "final_metrics.json")
    if (not args.force) and all(os.path.exists(p) for p in (final_non_path, final_bt_oracle_path, final_bt_rl_path)):
        print(f"[cache] {experiment.name}/{subset_label}: loading existing final metrics")
        with open(final_non_path, "r", encoding="utf-8") as fp:
            non = json.load(fp)
        with open(final_bt_oracle_path, "r", encoding="utf-8") as fp:
            bt_oracle = json.load(fp)
        with open(final_bt_rl_path, "r", encoding="utf-8") as fp:
            bt_rl = json.load(fp)
        return non, bt_oracle, bt_rl

    ensure_dir(out_dir_non_bt); ensure_dir(out_dir_bt_oracle); ensure_dir(out_dir_bt_rl)
    problems = load_subset(args, subset_tsv)
    print(f"[run] {experiment.name}/{subset_label}: {len(problems)} rows from {subset_tsv}")

    attempts_fp, baseline_fp, attempts_writer, baseline_writer, attempts_tsv, baseline_tsv, summary_jsonl, final_metrics_json = open_writers(out_dir_non_bt, experiment, args.force)
    resume_state = load_resume_state(summary_jsonl, experiment.name) if not args.force else load_resume_state("/path/that/does/not/exist", experiment.name)
    completed_ids = set(resume_state["completed_ids"])
    if completed_ids:
        print(f"[resume] {experiment.name}/{subset_label}: found {len(completed_ids)} completed rows; continuing")

    nb_passed = int(resume_state["nb_passed"]); nb_total = int(resume_state["nb_total"]); nb_target = int(resume_state["nb_target"])
    bt_oracle_passed = int(resume_state["bt_oracle_passed"]); bt_oracle_total = int(resume_state["bt_oracle_total"]); bt_oracle_target = int(resume_state["bt_oracle_target"])
    bt_rl_passed = int(resume_state["bt_rl_passed"]); bt_rl_total = int(resume_state["bt_rl_total"]); bt_rl_target = int(resume_state["bt_rl_target"])
    evaluated = int(resume_state["evaluated"]); skipped = int(resume_state["skipped"])
    nb_counts: Dict[str, int] = dict(resume_state["nb_counts"])
    bt_oracle_counts: Dict[str, int] = dict(resume_state["bt_oracle_counts"])
    bt_rl_counts: Dict[str, int] = dict(resume_state["bt_rl_counts"])
    pending: List[Dict[str, Any]] = []

    def write_result(item: Dict[str, Any], result: Dict[str, Any]) -> None:
        nonlocal nb_passed, nb_total, nb_target, bt_oracle_passed, bt_oracle_total, bt_oracle_target, bt_rl_passed, bt_rl_total, bt_rl_target, evaluated
        evaluated += 1
        nb_eval = result["nonbacktracking_eval"]
        bt_oracle_eval = result["backtracking_with_oracle_tests_eval"]
        bt_rl_eval = result["backtracking_with_rl_generated_tests_eval"]
        nb_passed += int(nb_eval["full_suite_passed"]); nb_total += int(nb_eval["full_suite_total"]); nb_target += int(bool(nb_eval.get("target_ok", False)))
        bt_oracle_passed += int(bt_oracle_eval["full_suite_passed"]); bt_oracle_total += int(bt_oracle_eval["full_suite_total"]); bt_oracle_target += int(bool(bt_oracle_eval.get("target_ok", False)))
        bt_rl_passed += int(bt_rl_eval["full_suite_passed"]); bt_rl_total += int(bt_rl_eval["full_suite_total"]); bt_rl_target += int(bool(bt_rl_eval.get("target_ok", False)))
        _add_count(nb_counts, result["selected_source"])
        _add_count(bt_oracle_counts, result["backtracking_with_oracle_tests_source"])
        _add_count(bt_rl_counts, result["backtracking_with_rl_generated_tests_source"])
        debug_eval_row = result.get("debug_eval") or {}
        gen_eval_row = result.get("gen_eval") or {}
        debug_verifier_eval_row = result.get("debug_verifier_eval") or {}
        gen_verifier_eval_row = result.get("gen_verifier_eval") or {}
        rl_bt_base_v_row = result.get("rl_backtracking_baseline_verifier_eval") or {}
        rl_bt_selected_v_row = result.get("rl_backtracking_selected_verifier_eval") or {}
        attempts_writer.writerow([
            experiment.name, item["task_id"], item["question_id"], item["aidx"],
            item["baseline_passed"], item["baseline_total"], f"{item['baseline_acc_pct']:.2f}",
            debug_eval_row.get("full_suite_passed", ""), debug_eval_row.get("full_suite_total", ""), f"{debug_eval_row.get('full_suite_acc_pct', 0.0):.2f}" if debug_eval_row else "",
            gen_eval_row.get("full_suite_passed", ""), gen_eval_row.get("full_suite_total", ""), f"{gen_eval_row.get('full_suite_acc_pct', 0.0):.2f}" if gen_eval_row else "",
            debug_verifier_eval_row.get("full_suite_passed", ""), debug_verifier_eval_row.get("full_suite_total", ""),
            gen_verifier_eval_row.get("full_suite_passed", ""), gen_verifier_eval_row.get("full_suite_total", ""),
            rl_bt_base_v_row.get("full_suite_passed", ""), rl_bt_base_v_row.get("full_suite_total", ""),
            rl_bt_selected_v_row.get("full_suite_passed", ""), rl_bt_selected_v_row.get("full_suite_total", ""),
            result["selected_source"], nb_eval["full_suite_passed"], nb_eval["full_suite_total"], f"{nb_eval['full_suite_acc_pct']:.2f}",
            result["backtracking_with_oracle_tests_source"], bt_oracle_eval["full_suite_passed"], bt_oracle_eval["full_suite_total"], f"{bt_oracle_eval['full_suite_acc_pct']:.2f}",
            result["backtracking_with_rl_generated_tests_source"], bt_rl_eval["full_suite_passed"], bt_rl_eval["full_suite_total"], f"{bt_rl_eval['full_suite_acc_pct']:.2f}",
            int(bool(nb_eval.get("target_ok", False))), result.get("debug_prompt_chars", 0), result.get("gen_prompt_chars", 0), result.get("debug_output_chars", 0), result.get("gen_output_chars", 0),
            str(resolve_entry_point(item["entry_point"], result.get("selected_code", item["current_code"]))), result.get("note", item.get("note", "")),
            result.get("debug_code", ""), result.get("gen_code", ""), result.get("selected_code", ""),
        ])
        attempts_fp.flush()
        save_jsonl_line(summary_jsonl, {
            "event": "final_metrics", "experiment": experiment.name, "task_id": item["task_id"], "question_id": item["question_id"],
            "selected_assert_index": item["aidx"], "baseline_full_suite_passed": int(item["baseline_passed"]), "baseline_full_suite_total": int(item["baseline_total"]),
            "selected_source": result["selected_source"], "nonbacktracking_eval": {k: v for k, v in nb_eval.items() if k not in ("results",)},
            "backtracking_with_oracle_tests_source": result["backtracking_with_oracle_tests_source"], "backtracking_with_oracle_tests_eval": {k: v for k, v in bt_oracle_eval.items() if k not in ("results",)},
            "backtracking_with_rl_generated_tests_source": result["backtracking_with_rl_generated_tests_source"], "backtracking_with_rl_generated_tests_eval": {k: v for k, v in bt_rl_eval.items() if k not in ("results",)},
            "rl_backtracking_baseline_verifier_eval": result.get("rl_backtracking_baseline_verifier_eval"), "rl_backtracking_selected_verifier_eval": result.get("rl_backtracking_selected_verifier_eval"),
            "debug_eval": result.get("debug_eval"), "gen_eval": result.get("gen_eval"), "debug_verifier_eval": result.get("debug_verifier_eval"), "gen_verifier_eval": result.get("gen_verifier_eval"),
            "candidate_evals": result.get("candidate_evals"), "candidate_codes": result.get("candidate_codes"),
            "selected_sample_index": result.get("selected_sample_index"),
            "note": result.get("note", item.get("note", "")),
        })
        completed_ids.add(_row_key(item["task_id"], item["question_id"]))

    def handle_skipped(item: Dict[str, Any], reason: str) -> None:
        nonlocal skipped
        skipped += 1
        baseline_eval = _baseline_eval_from_item(item)
        write_result(item, {"selected_source": "buggy", "selected_code": item["current_code"], "nonbacktracking_eval": baseline_eval, "backtracking_with_oracle_tests_source": "buggy", "backtracking_with_oracle_tests_eval": baseline_eval, "backtracking_with_rl_generated_tests_source": "buggy", "backtracking_with_rl_generated_tests_eval": baseline_eval, "note": reason})

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        need_debug = experiment.kind in ("single_debug", "best2_debug_oracle", "best2_debug_verifier", "routing_oracle", "routing_verifier")
        need_gen = experiment.kind in ("single_generation", "best2_generation_oracle", "best2_generation_verifier", "routing_oracle", "routing_verifier")
        best2_debug = experiment.kind in ("best2_debug_oracle", "best2_debug_verifier")
        best2_gen = experiment.kind in ("best2_generation_oracle", "best2_generation_verifier")
        debug_outs = []
        gen_outs = []
        if need_debug:
            assert debug_runner is not None
            if best2_debug:
                debug_msgs = []
                for x in pending:
                    msg = build_debug_messages(x["current_code"], x["assert_text"])
                    debug_msgs.extend([msg, msg])
            else:
                debug_msgs = [build_debug_messages(x["current_code"], x["assert_text"]) for x in pending]
            debug_outs = debug_runner.generate_batch(debug_msgs, args.deterministic_test, args.max_new_tokens, args.temperature, args.top_p)
        if need_gen:
            assert generation_runner is not None
            if best2_gen:
                gen_msgs = []
                for x in pending:
                    msg = build_generation_messages(x["generation_query"])
                    gen_msgs.extend([msg, msg])
            else:
                gen_msgs = [build_generation_messages(x["generation_query"]) for x in pending]
            gen_outs = generation_runner.generate_batch(gen_msgs, args.deterministic_test, args.max_new_tokens, args.temperature, args.top_p)
        for idx, item in enumerate(pending):
            if best2_debug:
                d_slice = debug_outs[2 * idx: 2 * idx + 2]
                debug_raws = [x[0] for x in d_slice]
                debug_prompt_len = d_slice[0][1] if d_slice else 0
            else:
                debug_raws = [debug_outs[idx][0]] if need_debug else []
                debug_prompt_len = debug_outs[idx][1] if need_debug else 0
            if best2_gen:
                g_slice = gen_outs[2 * idx: 2 * idx + 2]
                gen_raws = [x[0] for x in g_slice]
                gen_prompt_len = g_slice[0][1] if g_slice else 0
            else:
                gen_raws = [gen_outs[idx][0]] if need_gen else []
                gen_prompt_len = gen_outs[idx][1] if need_gen else 0

            debug_codes = [(extract_fenced_code(x) or x) for x in debug_raws]
            gen_codes = [(extract_generation_code(x) or x) for x in gen_raws]
            debug_raw = debug_raws[0] if debug_raws else ""
            gen_raw = gen_raws[0] if gen_raws else ""
            debug_code = debug_codes[0] if debug_codes else ""
            gen_code = gen_codes[0] if gen_codes else ""
            debug_eval = evaluate_candidate(item["query_text"], debug_code, item["test_code"], item["entry_point"], item["aidx"]) if need_debug else None
            gen_eval = evaluate_candidate(item["query_text"], gen_code, item["test_code"], item["entry_point"], item["aidx"]) if need_gen else None
            baseline_eval = _baseline_eval_from_item(item)
            selected_source = "buggy"; selected_code = item["current_code"]; selected_eval = baseline_eval
            debug_verifier_eval = None; gen_verifier_eval = None
            note = item.get("note", "")
            candidate_evals = None
            candidate_codes = None
            selected_sample_index = None
            if experiment.kind == "single_debug":
                selected_source, selected_code, selected_eval = "debug", debug_code, debug_eval
            elif experiment.kind == "single_generation":
                selected_source, selected_code, selected_eval = "generation", gen_code, gen_eval
            elif experiment.kind == "best2_generation_oracle":
                gen_evals = [evaluate_candidate(item["query_text"], code, item["test_code"], item["entry_point"], item["aidx"]) for code in gen_codes]
                selected_sample_index = choose_best_oracle_candidate(gen_evals)
                selected_source = f"generation_sample_{selected_sample_index + 1}"
                selected_code = gen_codes[selected_sample_index]
                selected_eval = gen_evals[selected_sample_index]
                gen_eval = selected_eval
                gen_code = selected_code
                gen_raw = gen_raws[selected_sample_index]
                candidate_evals = gen_evals
                candidate_codes = gen_codes
                note = (note + ";" if note else "") + summarize_candidate_evals("gen", gen_evals, selected_sample_index)
            elif experiment.kind == "best2_debug_oracle":
                debug_evals = [evaluate_candidate(item["query_text"], code, item["test_code"], item["entry_point"], item["aidx"]) for code in debug_codes]
                selected_sample_index = choose_best_oracle_candidate(debug_evals)
                selected_source = f"debug_sample_{selected_sample_index + 1}"
                selected_code = debug_codes[selected_sample_index]
                selected_eval = debug_evals[selected_sample_index]
                debug_eval = selected_eval
                debug_code = selected_code
                debug_raw = debug_raws[selected_sample_index]
                candidate_evals = debug_evals
                candidate_codes = debug_codes
                note = (note + ";" if note else "") + summarize_candidate_evals("debug", debug_evals, selected_sample_index)
            elif experiment.kind == "best2_generation_verifier":
                verifier_test_code = item["verifier_test_code"]
                verifier_aidx = int(item["verifier_aidx"])
                gen_evals_official = [evaluate_candidate(item["query_text"], code, item["test_code"], item["entry_point"], item["aidx"]) for code in gen_codes]
                gen_evals_verifier = [evaluate_candidate(item["query_text"], code, verifier_test_code, item["entry_point"], verifier_aidx) for code in gen_codes]
                selected_sample_index = choose_best_oracle_candidate(gen_evals_verifier)
                selected_source = f"generation_sample_{selected_sample_index + 1}"
                selected_code = gen_codes[selected_sample_index]
                selected_eval = gen_evals_official[selected_sample_index]
                gen_eval = selected_eval
                gen_verifier_eval = gen_evals_verifier[selected_sample_index]
                gen_code = selected_code
                gen_raw = gen_raws[selected_sample_index]
                candidate_evals = gen_evals_official
                candidate_codes = gen_codes
                note = (note + ";" if note else "") + "verifier_selection;" + summarize_candidate_evals("verifier_gen", gen_evals_verifier, selected_sample_index) + ";official_" + summarize_candidate_evals("gen", gen_evals_official, selected_sample_index)
            elif experiment.kind == "best2_debug_verifier":
                verifier_test_code = item["verifier_test_code"]
                verifier_aidx = int(item["verifier_aidx"])
                debug_evals_official = [evaluate_candidate(item["query_text"], code, item["test_code"], item["entry_point"], item["aidx"]) for code in debug_codes]
                debug_evals_verifier = [evaluate_candidate(item["query_text"], code, verifier_test_code, item["entry_point"], verifier_aidx) for code in debug_codes]
                selected_sample_index = choose_best_oracle_candidate(debug_evals_verifier)
                selected_source = f"debug_sample_{selected_sample_index + 1}"
                selected_code = debug_codes[selected_sample_index]
                selected_eval = debug_evals_official[selected_sample_index]
                debug_eval = selected_eval
                debug_verifier_eval = debug_evals_verifier[selected_sample_index]
                debug_code = selected_code
                debug_raw = debug_raws[selected_sample_index]
                candidate_evals = debug_evals_official
                candidate_codes = debug_codes
                note = (note + ";" if note else "") + "verifier_selection;" + summarize_candidate_evals("verifier_debug", debug_evals_verifier, selected_sample_index) + ";official_" + summarize_candidate_evals("debug", debug_evals_official, selected_sample_index)
            elif experiment.kind == "routing_oracle":
                assert debug_eval is not None and gen_eval is not None
                selected_source = choose_candidate(debug_eval, gen_eval)
                selected_code, selected_eval = (debug_code, debug_eval) if selected_source == "debug" else (gen_code, gen_eval)
            elif experiment.kind == "routing_verifier":
                verifier_test_code = item["verifier_test_code"]
                verifier_aidx = int(item["verifier_aidx"])
                debug_verifier_eval = evaluate_candidate(item["query_text"], debug_code, verifier_test_code, item["entry_point"], verifier_aidx)
                gen_verifier_eval = evaluate_candidate(item["query_text"], gen_code, verifier_test_code, item["entry_point"], verifier_aidx)
                selected_source = choose_candidate(debug_verifier_eval, gen_verifier_eval)
                selected_code = debug_code if selected_source == "debug" else gen_code
                selected_eval = evaluate_candidate(item["query_text"], selected_code, item["test_code"], item["entry_point"], item["aidx"])
            if selected_eval["full_suite_passed"] > item["baseline_passed"]:
                bt_oracle_source, bt_oracle_eval = selected_source, selected_eval
            else:
                bt_oracle_source, bt_oracle_eval = "buggy", baseline_eval
            bt_rl_source, bt_rl_eval, rl_bt_base_v_eval, rl_bt_selected_v_eval, note = _compute_rl_generated_backtracking(item, subset_tsv, selected_source, selected_code, selected_eval, baseline_eval, rl_backtracking_verifier_map, note)
            write_result(item, {"selected_source": selected_source, "selected_code": selected_code, "nonbacktracking_eval": selected_eval, "backtracking_with_oracle_tests_source": bt_oracle_source, "backtracking_with_oracle_tests_eval": bt_oracle_eval, "backtracking_with_rl_generated_tests_source": bt_rl_source, "backtracking_with_rl_generated_tests_eval": bt_rl_eval, "rl_backtracking_baseline_verifier_eval": rl_bt_base_v_eval, "rl_backtracking_selected_verifier_eval": rl_bt_selected_v_eval, "debug_eval": debug_eval, "gen_eval": gen_eval, "debug_verifier_eval": debug_verifier_eval, "gen_verifier_eval": gen_verifier_eval, "candidate_evals": candidate_evals, "candidate_codes": candidate_codes, "selected_sample_index": selected_sample_index, "debug_code": debug_code, "gen_code": gen_code, "debug_prompt_chars": debug_prompt_len, "gen_prompt_chars": gen_prompt_len, "debug_output_chars": len(debug_raw), "gen_output_chars": len(gen_raw), "note": note})
        pending = []

    try:
        for _, row in tqdm(problems.iterrows(), total=len(problems), desc=f"[{experiment.name}] {subset_label}", mininterval=args.tqdm_mininterval, disable=args.disable_tqdm):
            raw_task_id = str(row.get("task_id", "")); raw_question_id = str(row.get("question_id", ""))
            if _row_key(raw_task_id, raw_question_id) in completed_ids:
                continue
            item = baseline_for_row(row)
            if _row_key(item["task_id"], item["question_id"]) in completed_ids:
                continue
            baseline_writer.writerow([item["task_id"], item["question_id"], item["aidx"], item["baseline_selected_assert_failing"], item["baseline_passed"], item["baseline_total"], f"{item['baseline_acc_pct']:.2f}", item.get("note", "")])
            baseline_fp.flush()
            if experiment.kind == "buggy":
                baseline_eval = _baseline_eval_from_item(item)
                write_result(item, {"selected_source": "buggy", "selected_code": item["current_code"], "nonbacktracking_eval": baseline_eval, "backtracking_with_oracle_tests_source": "buggy", "backtracking_with_oracle_tests_eval": baseline_eval, "backtracking_with_rl_generated_tests_source": "buggy", "backtracking_with_rl_generated_tests_eval": baseline_eval, "note": item.get("note", "")})
                continue
            if item.get("note") == "skipped_no_failing_asserts" and experiment.kind in ("single_debug", "best2_debug_oracle", "best2_debug_verifier", "routing_oracle", "routing_verifier"):
                handle_skipped(item, "skipped_no_failing_asserts")
                continue
            if experiment.kind in ("single_generation", "best2_generation_oracle", "best2_generation_verifier", "routing_oracle", "routing_verifier"):
                if leetcode_map is None:
                    raise RuntimeError("LeetCode dataset map was not loaded but generation is required")
                item["generation_query"] = get_generation_query(item["task_id"], leetcode_map)
                if not item["generation_query"]:
                    item["note"] = (item.get("note", "") + ";" if item.get("note") else "") + "missing_generation_query"
            if experiment.kind in ("routing_verifier", "best2_generation_verifier", "best2_debug_verifier"):
                assert verifier_map is not None
                vinfo = get_verifier_info(verifier_map, item["task_id"], item["question_id"], subset_tsv)
                if vinfo is None or not str(vinfo.get("verifier_test_code", "")).strip():
                    handle_skipped(item, "skipped_missing_or_empty_verifier")
                    continue
                item["verifier_test_code"] = str(vinfo["verifier_test_code"])
                item["verifier_num_asserts"] = int(vinfo.get("num_asserts", 0))
                item["verifier_aidx"] = _find_verifier_aidx(item["query_text"], item["current_code"], item["verifier_test_code"], item["entry_point"])
                if item["verifier_aidx"] == 0:
                    item["note"] = (item.get("note", "") + ";" if item.get("note") else "") + "verifier_index_0_used_for_routing"
            pending.append(item)
            if len(pending) >= args.batch_size:
                flush_pending()
        flush_pending()
    finally:
        attempts_fp.close(); baseline_fp.close()

    final_non = build_final_dict(experiment, "non_backtracking", subset_tsv, args.split_mode, len(problems), evaluated, skipped, nb_passed, nb_total, nb_target, nb_counts, args)
    final_bt_oracle = build_final_dict(experiment, "backtracking_with_oracle_tests", subset_tsv, args.split_mode, len(problems), evaluated, skipped, bt_oracle_passed, bt_oracle_total, bt_oracle_target, bt_oracle_counts, args)
    final_bt_rl = build_final_dict(experiment, "backtracking_with_rl_generated_tests", subset_tsv, args.split_mode, len(problems), evaluated, skipped, bt_rl_passed, bt_rl_total, bt_rl_target, bt_rl_counts, args)
    with open(final_metrics_json, "w", encoding="utf-8") as fp: json.dump(final_non, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir_bt_oracle, "final_metrics.json"), "w", encoding="utf-8") as fp: json.dump(final_bt_oracle, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir_bt_rl, "final_metrics.json"), "w", encoding="utf-8") as fp: json.dump(final_bt_rl, fp, ensure_ascii=False, indent=2)
    for out_dir in (out_dir_bt_oracle, out_dir_bt_rl):
        shutil.copy2(attempts_tsv, os.path.join(out_dir, os.path.basename(attempts_tsv)))
        shutil.copy2(baseline_tsv, os.path.join(out_dir, os.path.basename(baseline_tsv)))
        if os.path.exists(summary_jsonl): shutil.copy2(summary_jsonl, os.path.join(out_dir, os.path.basename(summary_jsonl)))
    return final_non, final_bt_oracle, final_bt_rl


def run_experiment(experiment: Experiment, args: argparse.Namespace, leetcode_map: Optional[Dict[Any, Dict[str, Any]]], debug_runner: Optional[ModelRunner], generation_runner: Optional[ModelRunner], verifier_map: Optional[Dict[str, Dict[str, Any]]], verifier_jsonl: Optional[str], rl_backtracking_verifier_map: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    root = os.path.join(args.log_dir, experiment.name, args.split_mode.upper())
    ensure_dir(root)
    split_configs = [(args.subset_tsv_easy, "Easy"), (args.subset_tsv_medium, "Medium"), (args.subset_tsv_hard, "Hard"), (args.subset_tsv_extra_hard, "Extra_Hard")]
    per_non: List[Dict[str, Any]] = []; per_bt_oracle: List[Dict[str, Any]] = []; per_bt_rl: List[Dict[str, Any]] = []
    for subset_tsv, label in split_configs:
        out_non = os.path.join(root, f"{label}_Inference")
        out_bt_oracle = os.path.join(root, f"{label}_Backtracking_With_Oracle_Tests")
        out_bt_rl = os.path.join(root, f"{label}_Backtracking_With_RL_Generated_Tests")
        non, bt_oracle, bt_rl = run_subset(experiment, subset_tsv, label, out_non, out_bt_oracle, out_bt_rl, args, leetcode_map, debug_runner, generation_runner, verifier_map, verifier_jsonl, rl_backtracking_verifier_map)
        per_non.append(non); per_bt_oracle.append(bt_oracle); per_bt_rl.append(bt_rl)
    final_all = {"experiment": experiment.name, "table_label": experiment.table_label, "kind": experiment.kind, "debug_model_role": experiment.debug_model, "generation_model_role": experiment.generation_model, "verifier_source": experiment.verifier_source, "verifier_jsonl": verifier_jsonl, "rl_backtracking_verifier_jsonl": args.rl_testgen_verifier_jsonl, "split_mode": args.split_mode, "non_backtracking": {"global": sum_final_metrics(experiment, per_non, "non_backtracking"), "per_subset": per_non}, "backtracking_with_oracle_tests": {"global": sum_final_metrics(experiment, per_bt_oracle, "backtracking_with_oracle_tests"), "per_subset": per_bt_oracle}, "backtracking_with_rl_generated_tests": {"global": sum_final_metrics(experiment, per_bt_rl, "backtracking_with_rl_generated_tests"), "per_subset": per_bt_rl}, "seed": int(args.seed), "batch_size": int(args.batch_size), "deterministic_generation": bool(args.deterministic_test)}
    final_all_path = os.path.join(root, "final_metrics_all.json")
    with open(final_all_path, "w", encoding="utf-8") as fp: json.dump(final_all, fp, ensure_ascii=False, indent=2)
    print(f"[done] {experiment.name}: nonBT={final_all['non_backtracking']['global']['overall_accuracy_pct']:.2f} BT_oracle={final_all['backtracking_with_oracle_tests']['global']['overall_accuracy_pct']:.2f} BT_RL_tests={final_all['backtracking_with_rl_generated_tests']['global']['overall_accuracy_pct']:.2f} -> {final_all_path}")
    return final_all


def write_master_summary(log_dir: str, results: List[Dict[str, Any]]) -> None:
    path_json = os.path.join(log_dir, "all_selected_experiments_summary.json")
    with open(path_json, "w", encoding="utf-8") as fp: json.dump(results, fp, ensure_ascii=False, indent=2)
    path_tsv = os.path.join(log_dir, "all_selected_experiments_summary.tsv")
    with open(path_tsv, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerow(["experiment", "table_label", "mode", "overall_accuracy_pct", "passed", "total", "evaluated", "source_counts"])
        for res in results:
            for mode in ["non_backtracking", "backtracking_with_oracle_tests", "backtracking_with_rl_generated_tests"]:
                g = res[mode]["global"]
                w.writerow([res["experiment"], res["table_label"], mode, f"{g['overall_accuracy_pct']:.4f}", g["final_full_suite_passed_total"], g["final_full_suite_total_asserts"], g.get("evaluated", ""), json.dumps(g.get("source_counts", {}), sort_keys=True)])
    print(f"[summary] wrote {path_json} and {path_tsv}")


# -------------------------
# CLI orchestration
# -------------------------
def resolve_peft_checkpoint(args: argparse.Namespace) -> Optional[str]:
    ckpt = os.path.expanduser(os.path.expandvars(args.rl_peft_checkpoint or ""))
    if not ckpt:
        return None
    if os.path.isabs(ckpt):
        return ckpt
    root = os.path.expanduser(os.path.expandvars(args.peft_model_root or ""))
    if not root:
        raise ValueError("--rl_peft_checkpoint is relative; pass --peft_model_root as well")
    return os.path.join(root, ckpt)


def required_model_specs(exp: Experiment) -> List[str]:
    specs = []
    for s in [exp.debug_model, exp.generation_model]:
        if s and s not in specs:
            specs.append(s)
    return specs


def verifier_path_for_experiment(exp: Experiment, args: argparse.Namespace) -> Optional[str]:
    if exp.kind not in ("routing_verifier", "best2_generation_verifier", "best2_debug_verifier"):
        return None
    if exp.verifier_source == "base_testgen":
        return args.base_testgen_verifier_jsonl
    if exp.verifier_source == "rl_testgen":
        return args.rl_testgen_verifier_jsonl
    raise ValueError(f"Unknown verifier source: {exp.verifier_source}")


def select_experiments(args: argparse.Namespace) -> List[Experiment]:
    if args.all:
        names = ALL_EXPERIMENT_ORDER
    else:
        names = args.experiments or []
    if not names:
        raise ValueError("Pass --all or at least one --experiments entry")
    bad = [n for n in names if n not in EXPERIMENTS]
    if bad:
        raise ValueError(f"Unknown experiment(s): {bad}. Valid: {sorted(EXPERIMENTS)}")
    return [EXPERIMENTS[n] for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified runner for LeetCode generation/debug table experiments.")
    parser.add_argument("--experiments", nargs="*", default=[], choices=sorted(EXPERIMENTS.keys()), help="Experiment rows to run. Use --all to run all table experiments.")
    parser.add_argument("--all", action="store_true", help="Run all known table experiments sequentially.")
    parser.add_argument("--log_dir", type=str, required=True, help="Output root directory.")

    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--generation_base_model_id", type=str, default="", help="Optional base model ID for generation. Defaults to --base_model_id.")
    parser.add_argument("--rl_peft_checkpoint", type=str, default=DEFAULT_RL_PEFT_CHECKPOINT, help="RL debugger PEFT checkpoint applied on top of --base_model_id. Defaults to the checkpoint-3000 bug-fixer adapter.")
    parser.add_argument("--peft_model_root", type=str, default="")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    parser.add_argument("--leetcode_dataset_name", type=str, default=DEFAULT_LEETCODE_DATASET_NAME)
    parser.add_argument("--leetcode_split", type=str, default=DEFAULT_LEETCODE_SPLIT)
    parser.add_argument("--subset_tsv_easy", type=str, default=DEFAULT_SUBSET_TSV_EASY)
    parser.add_argument("--subset_tsv_medium", type=str, default=DEFAULT_SUBSET_TSV_MEDIUM)
    parser.add_argument("--subset_tsv_hard", type=str, default=DEFAULT_SUBSET_TSV_HARD)
    parser.add_argument("--subset_tsv_extra_hard", type=str, default=DEFAULT_SUBSET_TSV_EXTRA_HARD)
    parser.add_argument("--split_mode", type=str, default="all", choices=["dev", "test", "all"])
    parser.add_argument("--limit_per_subset", type=int, default=0, help="Optional smoke-test limit per bucket after split. 0 means no limit.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start row index after split/limit filtering.")
    parser.add_argument("--end_idx", type=int, default=0, help="End row index exclusive after split/limit filtering. 0 means no end bound.")

    parser.add_argument("--base_testgen_verifier_jsonl", type=str, default=DEFAULT_BASE_TESTGEN_VERIFIER_JSONL, help="per_problem_detailed.jsonl from the base/untrained test generator.")
    parser.add_argument("--rl_testgen_verifier_jsonl", type=str, default=DEFAULT_RL_TESTGEN_VERIFIER_JSONL, help="per_problem_detailed.jsonl from the RL-trained test generator. Also used for backtracking_with_rl_generated_tests.")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--deterministic_test", action="store_true", help="Use greedy decoding. Default uses sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Recompute even if final_metrics.json already exists.")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--tqdm_mininterval", type=float, default=5.0)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    ensure_dir(args.log_dir)
    set_global_seeds(args.seed)

    experiments = select_experiments(args)
    needs_generation = any(e.kind in ("single_generation", "best2_generation_oracle", "best2_generation_verifier", "routing_oracle", "routing_verifier") for e in experiments)
    leetcode_map = None
    if needs_generation:
        print(f"[dataset] loading {args.leetcode_dataset_name} split={args.leetcode_split}")
        leetcode_map = load_leetcode_query_map(args.leetcode_dataset_name, args.leetcode_split)

    rl_backtracking_verifier_map = None
    if args.rl_testgen_verifier_jsonl:
        print(f"[verifier] loading RL testgen for generated-test backtracking: {args.rl_testgen_verifier_jsonl}")
        rl_backtracking_verifier_map = load_verifier_tests_map(args.rl_testgen_verifier_jsonl)

    rl_ckpt = resolve_peft_checkpoint(args)
    if any("rl" in required_model_specs(e) for e in experiments):
        if not rl_ckpt:
            raise ValueError("At least one selected experiment needs the RL debugger; pass --rl_peft_checkpoint")
        if not os.path.isdir(rl_ckpt):
            raise FileNotFoundError(f"RL PEFT checkpoint not found: {rl_ckpt}")

    results: List[Dict[str, Any]] = []
    generation_base_model_id = args.generation_base_model_id or args.base_model_id

    for exp in experiments:
        print("\n" + "=" * 80)
        print(f"[experiment] {exp.name}: {exp.table_label}")
        print("=" * 80)

        verifier_map = None
        verifier_jsonl = verifier_path_for_experiment(exp, args)
        if exp.kind in ("routing_verifier", "best2_generation_verifier", "best2_debug_verifier"):
            if not verifier_jsonl:
                raise ValueError(f"{exp.name} needs verifier JSONL for source {exp.verifier_source}")
            print(f"[verifier] loading {verifier_jsonl}")
            verifier_map = load_verifier_tests_map(verifier_jsonl)

        debug_runner = None
        generation_runner = None
        specs = required_model_specs(exp)
        try:
            # Load the minimum set of models for this experiment. If both roles use the same spec, reuse one runner.
            loaded: Dict[str, ModelRunner] = {}
            for spec in specs:
                if spec == "base":
                    model_id = args.base_model_id
                    print(f"[model] loading base: {model_id}")
                    loaded[spec] = load_model_and_tokenizer(model_id, peft_checkpoint=None, dtype=args.dtype)
                elif spec == "rl":
                    assert rl_ckpt is not None
                    print(f"[model] loading RL debugger PEFT: base={args.base_model_id}, ckpt={rl_ckpt}")
                    loaded[spec] = load_model_and_tokenizer(args.base_model_id, peft_checkpoint=rl_ckpt, dtype=args.dtype)
                else:
                    raise ValueError(f"Unknown model spec: {spec}")

            # Optional: if generation role is base and user specified a different generation base model, reload for generation.
            if exp.generation_model == "base" and generation_base_model_id != args.base_model_id:
                if exp.debug_model == "base":
                    # The debug base and generation base differ, so keep debug base and load a separate generation base.
                    print(f"[model] loading separate generation base: {generation_base_model_id}")
                    generation_runner = load_model_and_tokenizer(generation_base_model_id, peft_checkpoint=None, dtype=args.dtype)
                    debug_runner = loaded.get(exp.debug_model) if exp.debug_model else None
                else:
                    print(f"[model] loading generation base: {generation_base_model_id}")
                    generation_runner = load_model_and_tokenizer(generation_base_model_id, peft_checkpoint=None, dtype=args.dtype)
                    debug_runner = loaded.get(exp.debug_model) if exp.debug_model else None
            else:
                debug_runner = loaded.get(exp.debug_model) if exp.debug_model else None
                generation_runner = loaded.get(exp.generation_model) if exp.generation_model else None

            res = run_experiment(exp, args, leetcode_map, debug_runner, generation_runner, verifier_map, verifier_jsonl, rl_backtracking_verifier_map)
            res["base_model_id"] = args.base_model_id
            res["generation_base_model_id"] = generation_base_model_id
            res["rl_base_model_id"] = args.base_model_id
            res["rl_peft_checkpoint"] = rl_ckpt
            results.append(res)
        finally:
            # Avoid double-free if same object reused.
            seen_ids = set()
            for r in [debug_runner, generation_runner]:
                if r is not None and id(r) not in seen_ids:
                    seen_ids.add(id(r)); free_runner(r)
            try:
                # Also free any runner loaded but not assigned due to separate-generation branch.
                for r in locals().get("loaded", {}).values():
                    if id(r) not in seen_ids:
                        seen_ids.add(id(r)); free_runner(r)
            except Exception:
                pass

    write_master_summary(args.log_dir, results)


if __name__ == "__main__":
    main()
