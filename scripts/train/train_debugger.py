#!/usr/bin/env python3

import os
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import sys, re, json, textwrap, logging, random, math, ast, argparse
from typing import Any, Dict, List, Tuple, Optional
import importlib.util
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

import wandb
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

# TRL imports (shim)
try:
    from trl import GRPOTrainer, GRPOConfig
except Exception:
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
        from trl.trainer.grpo_config import GRPOConfig
    except Exception as e:
        raise ImportError(
            "GRPOTrainer not found in your installed 'trl'. "
            "Please upgrade to TRL >= 0.9.x. Original error: " + str(e)
        )

from peft import LoraConfig

SCRIPT_PATH = os.path.abspath(__file__)
LOG_CONTEXT = {"script_path": SCRIPT_PATH}

CACHE_ROOT = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
HF_DATASETS_CACHE  = os.environ.get("HF_DATASETS_CACHE",  os.path.join(CACHE_ROOT, "datasets"))
CACHE_DIR = TRANSFORMERS_CACHE

# =========================
# PATHS & HYPERPARAMETERS
# =========================
OUTPUT_DIR = str(_REPO_ROOT / "checkpoints/debugger")

CURRICULUM_JSONL = os.path.join(OUTPUT_DIR, "training_unitcases_plan.jsonl")
CURRICULUM_STATS = os.path.join(OUTPUT_DIR, "training_unitcases_stats.json")
LOG_JSONL = os.path.join(OUTPUT_DIR, "training_curriculum_targeted_logs_train_Qwen3_8B_random_split.jsonl")
WANDB_RUN_ID_FILE = os.path.join(OUTPUT_DIR, "wandb_run_id.txt")

MODEL_ID = "Qwen/Qwen3-8B"
INPUT_TSV = str(_REPO_ROOT / "data/leetcode/train/debugger_train.tsv")
EVAL_FRACTION = 0.0

# NUM_GENERATIONS = 8
NUM_GENERATIONS = 4
# BATCH_SIZE = 2
BATCH_SIZE = 1
EPOCHS = 1
# MAX_PROMPT_TOKENS = 2048
MAX_PROMPT_TOKENS = 2048
MAX_COMPLETION_TOKENS = 1024

# Thinking-mode sampling (avoid greedy)
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 0.0

BETA = 0.02
LR = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
GRAD_ACCUM_STEPS = 2
CLIP_GRAD_NORM = 1.0

LOG_EVERY = 10
TIME_LIMIT_SEC = 10
SEED = 1012

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

TARGET_UNITCASE_SAMPLES = 3000
MAX_UNITCASES_PER_PROBLEM = 10

SAVE_EVERY_STEPS = 600
SAVE_TOTAL_LIMIT = 50

# =========================
# RANK / LOGGING
# =========================
def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# =========================
# tester.py (import)
# =========================
TESTER_PATH = str(_REPO_ROOT / "patssel/evaluation/leetcode.py")

def import_tester(path: str):
    spec = importlib.util.spec_from_file_location("tester", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import tester module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    req = ["evaluate_code", "eval_with_timeout", "eval_all_asserts_with_output",
           "eval_single_assert_with_output", "extract_assert_texts", "resolve_entry_point"]
    for k in req:
        if not hasattr(mod, k):
            raise RuntimeError(f"tester.py must define {k}")
    return mod

_tester = None
evaluate_code = eval_with_timeout = eval_all_asserts_with_output = None
eval_single_assert_with_output = extract_assert_texts = resolve_entry_point = None

# =========================
# GLOBAL BUFFERS
# =========================
CURRENT_EPOCH = 0
EPOCH_BUF = {
    "rewards": [],
    "fixed_flags": [],
    "baseline_failed": [],

    "reward_base": [],
    "reward_targeted_bonus": [],
    "reward_suite_extra": [],

    "pass_rate_before": [],
    "pass_rate_after": [],
    "delta_pass_rate": [],
    "frac_improve": [],
    "frac_regress": [],
    "additional_fixed": [],
    "additional_failed": [],
    "reward_suite_signed": [],

    "lev_bug_fix_norm": [],
    "ast_bug_fix_norm": [],
}
EVAL_ROWS: List[Dict[str, Any]] = []

# =========================
# CALLBACKS
# =========================
class EpochTrackerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        global CURRENT_EPOCH
        if state.epoch is not None:
            try:
                CURRENT_EPOCH = int(state.epoch)
            except Exception:
                try:
                    CURRENT_EPOCH = int(round(float(state.epoch)))
                except Exception:
                    pass

class WandbTrainingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process() or not logs:
            return
        passthrough = {
            f"train/{k}": v
            for k, v in logs.items()
            if any(s in k.lower() for s in ["entropy", "logp", "logps", "kl"])
        }
        if passthrough:
            wandb.log(passthrough)

    def on_step_end(self, args, state, control, **kwargs):
        if not is_main_process():
            return
        payload = {"epoch": state.epoch, "step": state.global_step}
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                payload["loss"] = last["loss"]
            if "learning_rate" in last:
                payload["learning_rate"] = last["learning_rate"]
        wandb.log(payload)

class EpochEndLoggerAndEvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not is_main_process():
            return

        if EPOCH_BUF["rewards"]:
            arr = np.array(EPOCH_BUF["rewards"], dtype=np.float32)
            base_fail = np.array(EPOCH_BUF["baseline_failed"], dtype=np.float32)
            fixed = (arr > 0).astype(np.float32)

            def _nanmean(key: str) -> float:
                x = np.array(EPOCH_BUF.get(key, []), dtype=np.float32)
                if x.size == 0:
                    return 0.0
                return float(np.nanmean(x))

            wandb.log({
                "epoch/reward_mean": float(arr.mean()),
                "epoch/reward_p95": float(np.percentile(arr, 95)),
                "epoch/fixed_rate": float(fixed.mean()),
                "epoch/base_failed_mean": float(base_fail.mean()) if base_fail.size else 0.0,
                "epoch/base_failed_std": float(base_fail.std()) if base_fail.size else 0.0,

                "epoch/reward_base_mean": _nanmean("reward_base"),
                "epoch/reward_targeted_bonus_mean": _nanmean("reward_targeted_bonus"),
                "epoch/reward_suite_extra_mean": _nanmean("reward_suite_extra"),

                "epoch/suite/pass_rate_before_mean": _nanmean("pass_rate_before"),
                "epoch/suite/pass_rate_after_mean": _nanmean("pass_rate_after"),
                "epoch/suite/delta_pass_rate_mean": _nanmean("delta_pass_rate"),
                "epoch/suite/frac_improve_mean": _nanmean("frac_improve"),
                "epoch/suite/frac_regress_mean": _nanmean("frac_regress"),
                "epoch/suite/additional_fixed_mean": _nanmean("additional_fixed"),
                "epoch/suite/additional_failed_mean": _nanmean("additional_failed"),
                "epoch/suite/reward_suite_signed_mean": _nanmean("reward_suite_signed"),

                "epoch/edit/lev_bug_fix_norm_mean": _nanmean("lev_bug_fix_norm"),
                "epoch/edit/ast_bug_fix_norm_mean": _nanmean("ast_bug_fix_norm"),

                "epoch": CURRENT_EPOCH,
            })
            try:
                wandb.log({"epoch/reward_hist": wandb.Histogram(arr, num_bins=20)})
            except Exception:
                pass

        for k in EPOCH_BUF:
            EPOCH_BUF[k].clear()

        if tokenizer is not None and model is not None and EVAL_ROWS:
            eval_metrics = run_heldout_eval(model, tokenizer, EVAL_ROWS)
            eval_metrics["epoch"] = CURRENT_EPOCH
            wandb.log(eval_metrics)

# =========================
# EXTRACTION HELPERS (token-slice first; then strip thinking; then last python fence)
# =========================
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
FENCE_PY_RE = re.compile(r"```python\s*([\s\S]*?)\s*```", re.IGNORECASE)

def _strip_think_blocks(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text or "").strip()

def _strip_prompt_echo_textwise(text: str, prompt: str) -> str:
    if not text:
        return ""
    if prompt and text.startswith(prompt):
        return text[len(prompt):]
    return text

def extract_generated_only_text(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    completion_ids: Optional[Any] = None,
    *,
    max_prompt_tokens: Optional[int] = None,
    think_token_id: Optional[int] = None,
    end_think_token_id: Optional[int] = 151668,  # Qwen3 example uses 151668 as </think>
) -> str:
    # compute prompt len in tokens (what model saw)
    try:
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        if max_prompt_tokens is not None:
            prompt_len = min(prompt_len, int(max_prompt_tokens))
    except Exception:
        prompt_len = None

    # normalize ids
    ids: Optional[List[int]] = None
    if completion_ids is not None:
        try:
            if hasattr(completion_ids, "detach"):
                ids = completion_ids.detach().cpu().tolist()
            else:
                ids = completion_ids

            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
                ids = ids[0]
        except Exception:
            ids = None

    # token path
    if isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids):
        gen_ids = ids

        # decide if full sequence
        if prompt_len is not None and len(ids) > prompt_len:
            try:
                prefix_txt = tokenizer.decode(ids[:prompt_len], skip_special_tokens=True)
                # loose sanity check, then slice anyway
                if prefix_txt.strip():
                    gen_ids = ids[prompt_len:]
                else:
                    gen_ids = ids[prompt_len:]
            except Exception:
                gen_ids = ids[prompt_len:]

        # optional: remove think tokens via </think> id if present
        if end_think_token_id is not None:
            try:
                if end_think_token_id in gen_ids:
                    idx = len(gen_ids) - gen_ids[::-1].index(end_think_token_id)
                    gen_ids = gen_ids[idx:]
            except Exception:
                pass

        try:
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip("\n")
        except Exception:
            gen_text = (completion_text or "").strip()

        return _strip_think_blocks(gen_text)

    # fallback heuristics (text-only)
    gen_text = _strip_prompt_echo_textwise((completion_text or ""), prompt_text)
    gen_text = _strip_think_blocks(gen_text)
    return gen_text.strip()

def extract_last_python_fence(text: str) -> str:
    matches = FENCE_PY_RE.findall(text or "")
    return matches[-1].strip() if matches else ""

def looks_like_function_or_class(code: str) -> bool:
    return bool(re.search(r"^\s*(class\s+\w+\s*:|def\s+\w+\s*\()", code or "", re.M))

def append_jsonl(obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(LOG_JSONL), exist_ok=True)
    payload = {**LOG_CONTEXT, **obj}
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ---- Levenshtein + AST distance helpers (logging only; reward unchanged) ----
def _levenshtein_seq(seq1, seq2) -> int:
    m, n = len(seq1), len(seq2)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        s_i = seq1[i - 1]
        for j in range(1, n + 1):
            t_j = seq2[j - 1]
            cost = 0 if s_i == t_j else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def levenshtein_str(a: str, b: str) -> int:
    return _levenshtein_seq(list(a or ""), list(b or ""))

def ast_node_sequence(code: str) -> List[str]:
    tree = ast.parse(code)
    seq: List[str] = []
    def visit(node):
        seq.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            visit(child)
    visit(tree)
    return seq

def ast_distance(code_a: str, code_b: str) -> Tuple[float, float]:
    try:
        seq_a = ast_node_sequence(code_a or "")
        seq_b = ast_node_sequence(code_b or "")
    except Exception:
        return float("nan"), float("nan")
    dist = _levenshtein_seq(seq_a, seq_b)
    norm = dist / float(max(1, len(seq_a)))
    return float(dist), float(norm)

def _pair_dists(a: str, b: str) -> Dict[str, float]:
    a = a or ""
    b = b or ""
    lev = float(levenshtein_str(a, b)) if (a or b) else 0.0
    max_len = float(max(1, max(len(a), len(b))))
    lev_norm = lev / max_len
    ast_d, ast_n = ast_distance(a, b)
    return {"lev": float(lev), "lev_norm": float(lev_norm), "ast": float(ast_d), "ast_norm": float(ast_n)}

# =========================
# NEW STRUCTURED PROMPT (few-shot included; uses <PYTHON> markers)
# =========================
SYSTEM_MSG = (
    "You are a Python bug-fixing assistant.\n\n"
    "You MUST respond with the exact structure below. Every section is REQUIRED.\n"
    "Write naturally and with as much detail as needed (no length limit).\n\n"
    "1) CODE INTENT\n"
    "   Describe what the function/class is intended to do, in plain English.\n\n"
    "2) BUG REVIEW (fill EVERY tag; write 'N/A' if none)\n"
    "   For each tag, write a clear natural-language diagnosis of the issue, including symptoms.\n"
    "   If a tag is N/A, briefly state why.\n\n"
    "   - Syntax Errors:\n"
    "     This category is for code that Python cannot parse or import (e.g., missing colon, unmatched bracket).\n\n"
    "   - Runtime Errors:\n"
    "     This category is for failures that happen during execution, such as exceptions, infinite loops, timeouts, or hangs.\n"
    "     Include the likely exception name or the non-termination symptom when applicable.\n\n"
    "   - Core Logic Bugs:\n"
    "     This category is for mistakes in the main algorithm or control flow that make typical inputs behave incorrectly.\n\n"
    "   - Functional Bugs (subtle / edge cases):\n"
    "     This category is for boundary conditions and tricky cases (small inputs, negatives, empty cases, type quirks, iterator misuse, off-by-one).\n"
    "     Include issues that may not appear on the most common tests but still violate expected behavior.\n\n"
    "   - Spec / Test Mismatch:\n"
    "     This category is for cases where the code's behavior contradicts the unit test's implied specification.\n"
    "     If the test is the ground truth, explain exactly how the current behavior diverges from it.\n\n"
    "3) FIX PLAN\n"
    "   Give a step-by-step plan for the minimal safe fix.\n"
    "   Mention what you will NOT change (e.g., keep signature, do not change imports).\n\n"
    "4) FINAL FIX\n"
    "   Output ONLY the fixed function/class in exactly ONE Python fenced code block (```python\n<function or class>\n```).\n"
    "   The code block MUST be the LAST thing in the response.\n"
    "   Do NOT include any other code fences anywhere else.\n"
    "   Keep the same function/class name and signature.\n"
    "   Do NOT change imports.\n\n"
    "-----------------\n"
    "FEW-SHOT EXAMPLE\n\n"
    "Input:\n"
    "Buggy code:\n"
    "<PYTHON>\n"
    "def is_palindrome(x: int) -> bool:\n"
    "    if x < 0: return True             \n"
    "    s = str(x)\n"
    "    if len(s) == 1: return False       \n"
    "    i, j = 0, len(s) - 1\n"
    "    while i <= j:                      \n"
    "        if s[i] != s[j]:\n"
    "            return True                \n"
    "        i += 1; j += 1                 \n"
    "    return s == reversed(s)            \n"
    "</PYTHON>\n\n"
    "Failing test case:\n"
    "assert is_palindrome(121) is True\n\n"
    "Output:\n"
    "1) CODE INTENT\n"
    "This function should return True if the integer reads the same forward and backward (a palindrome),\n"
    "and False otherwise. Negative numbers should not be considered palindromes.\n\n"
    "2) BUG REVIEW:\n\n"
    "- Syntax Errors:\n\n"
    "N/A. The function is syntactically valid and can be parsed and imported by Python.\n"
    "Meaning of N/A: this category applies only when Python cannot parse the code (e.g., missing colon, unmatched bracket). "
    "That is not the case here.\n\n"
    "- Runtime Errors:\n\n"
    "Present. The loop can fail to terminate.\n\n"
    "The update j += 1 moves the right pointer outward instead of inward.\n\n"
    "For inputs where s[i] == s[j] repeatedly (e.g., 121), the loop does not hit a return path and continues indefinitely.\n\n"
    "This constitutes a runtime failure via non-termination (timeout/hang), even if no exception is raised.\n\n"
    "- Core Logic Bugs:\n\n"
    "Present. The algorithm’s control decisions are inverted or structurally incorrect.\n\n"
    "Negative input handling is reversed: x < 0 returns True when it should return False.\n\n"
    "Single-digit handling is reversed: len(s) == 1 returns False when it should return True.\n\n"
    "Mismatch handling is reversed: s[i] != s[j] returns True when it should return False.\n\n"
    "Pointer movement is incorrect: j is incremented rather than decremented, preventing correct inward comparison.\n\n"
    "- Functional Bugs (subtle / edge cases):\n\n"
    "Present. Several boundary and representation-related failures exist.\n\n"
    "Single-digit inputs are incorrectly classified as non-palindromes due to the early return.\n\n"
    "The final line s == reversed(s) compares a string to a reverse-iterator, not to a reversed string. "
    "This yields an incorrect result even if earlier logic were repaired.\n\n"
    "- Spec / Test Mismatch:\n\n"
    "Present. The implementation behavior diverges from the standard “Palindrome Number” specification.\n\n"
    "The spec expects palindromes like 121 to return True; the function can hang instead of returning.\n\n"
    "3) FIX PLAN:\n\n"
    "Negatives will be made to return False (instead of True).\n\n"
    "The incorrect single-digit early return will be removed so single digits evaluate to True.\n\n"
    "The loop condition will be changed to i < j (instead of i <= j).\n\n"
    "On a digit mismatch, the function will return False (instead of True).\n\n"
    "The right pointer will be decremented with j -= 1 (instead of incremented), preventing non-termination.\n\n"
    "The invalid final comparison s == reversed(s) will be removed and the function will return True after all symmetric checks pass.\n\n"
    "4) FINAL FIX:\n"
    "```python\n"
    "def is_palindrome(x: int) -> bool:\n"
    "    if x < 0:\n"
    "        return False\n"
    "    s = str(x)\n"
    "    i, j = 0, len(s) - 1\n"
    "    while i < j:\n"
    "        if s[i] != s[j]:\n"
    "            return False\n"
    "        i += 1\n"
    "        j -= 1\n"
    "    return True\n"
    "```\n"
)

def build_targeted_prompt(buggy_code: str, assert_text: str) -> str:
    return (
        "Fix the buggy function/class so that the failing unit test passes, also try to fix any other issues in the code.\n"
        "- Keep the same function/class name and signature.\n"
        "- Do NOT change imports.\n"
        "- In your response, the FINAL FIX section must end with exactly ONE Python code fence like:\n"
        "```python\n<function or class>\n```\n\n"
        "Buggy function/class:\n"
        f"{textwrap.dedent(buggy_code).strip()}\n\n"
        "Failing unit test:\n"
        f"{assert_text}\n"
    )

def qwen_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

# =========================
# DATA PREP: Build targeted plan
# =========================
def load_buggy_pool() -> pd.DataFrame:
    df = pd.read_csv(INPUT_TSV, sep="\t", dtype={"task_id": str, "question_id": str})
    req = {"task_id", "question_id", "prediction"}
    if not req.issubset(df.columns):
        raise ValueError(f"Input TSV must have columns {sorted(req)}")
    return df

def attach_meta(df: pd.DataFrame) -> pd.DataFrame:
    ds = load_dataset("newfacade/LeetCodeDataset", cache_dir=HF_DATASETS_CACHE)
    meta_rows: List[Dict[str, Any]] = []
    for split in ds.keys():
        for ex in ds[split]:
            meta_rows.append({
                "task_id": str(ex.get("task_id", "")),
                "question_id": str(ex.get("question_id", "")),
                "query": ex.get("query", "") or ex.get("prompt", ""),
                "test_code": ex.get("test", "") or ex.get("test_code", ""),
                "entry_point": ex.get("entry_point", None),
                "gold_completion": "" if ex.get("completion", None) is None else str(ex.get("completion", "")),
            })
    meta = pd.DataFrame(meta_rows).drop_duplicates(subset=["task_id", "question_id"], keep="first")
    m = df.merge(meta, on=["task_id", "question_id"], how="left")
    m = m[m["test_code"].astype(str).str.len() > 0].copy()
    return m

def evaluate_buggy_and_get_fails(row: pd.Series) -> Tuple[Dict[str, Any], List[int], List[str]]:
    query_text = row.get("query", "") or ""
    buggy_code = str(row["prediction"] or "")
    test_code = str(row["test_code"])
    entry_point = row.get("entry_point", None)

    allres = eval_all_asserts_with_output(query_text, buggy_code, test_code, entry_point)
    assert_texts = extract_assert_texts(test_code)
    failing = [r["index"] for r in allres.get("results", []) if not r.get("ok")]
    total = int(allres.get("total", len(assert_texts)))
    passed = total - len(failing)

    base = {
        "baseline_total": total,
        "baseline_failed": len(failing),
        "baseline_passed": passed,
        "failing_indices": failing,
    }
    return base, failing, assert_texts

def build_training_plan(tokenizer: AutoTokenizer) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(SEED)
    pool = attach_meta(load_buggy_pool())

    idxs = np.arange(len(pool))
    rng.shuffle(idxs)
    pool = pool.iloc[idxs].reset_index(drop=True)

    plan: List[Dict[str, Any]] = []
    per_problem_counts: Dict[Tuple[str, str], int] = {}

    for _, row in pool.iterrows():
        tid = row["task_id"]; qid = row["question_id"]
        key = (tid, qid)

        if per_problem_counts.get(key, 0) >= MAX_UNITCASES_PER_PROBLEM:
            continue

        base, failing, assert_texts = evaluate_buggy_and_get_fails(row)
        if base["baseline_failed"] <= 0:
            continue

        local = failing[:]
        rng.shuffle(local)
        quota_left = MAX_UNITCASES_PER_PROBLEM - per_problem_counts.get(key, 0)
        take = local[:quota_left]

        for aidx in take:
            assert_text = assert_texts[aidx] if aidx < len(assert_texts) else f"assert #{aidx}"
            targeted_user_text = build_targeted_prompt(row["prediction"], assert_text)
            chat_prompt = qwen_chat_prompt(tokenizer, targeted_user_text)
            prompt_token_len = len(tokenizer(chat_prompt, add_special_tokens=False).input_ids)

            plan.append({
                "prompt": chat_prompt,
                "prompt_token_len": int(prompt_token_len),

                "task_id": tid,
                "question_id": qid,
                "buggy_code": str(row["prediction"] or ""),

                "test_code": str(row["test_code"]),
                "entry_point": row.get("entry_point", None),
                "prompt_text": row.get("query", "") or "",
                "target_assert_index": int(aidx),
                "target_assert_text": assert_text,

                "baseline_total": int(base["baseline_total"]),
                "baseline_failed": int(base["baseline_failed"]),
                "baseline_passed": int(base["baseline_passed"]),
                "baseline_failing_indices": list(map(int, base["failing_indices"])),

                "gold_completion": str(row.get("gold_completion", "") or ""),
            })

            per_problem_counts[key] = per_problem_counts.get(key, 0) + 1
            if len(plan) >= TARGET_UNITCASE_SAMPLES:
                break
        if len(plan) >= TARGET_UNITCASE_SAMPLES:
            break

    if len(plan) < TARGET_UNITCASE_SAMPLES:
        for _, row in pool.iterrows():
            if len(plan) >= TARGET_UNITCASE_SAMPLES:
                break
            tid = row["task_id"]; qid = row["question_id"]
            base, failing, assert_texts = evaluate_buggy_and_get_fails(row)

            used_for_problem = {r["target_assert_index"] for r in plan if r["task_id"] == tid and r["question_id"] == qid}
            unused = [i for i in failing if i not in used_for_problem]
            rng.shuffle(unused)

            for aidx in unused:
                if len(plan) >= TARGET_UNITCASE_SAMPLES:
                    break
                assert_text = assert_texts[aidx] if aidx < len(assert_texts) else f"assert #{aidx}"
                targeted_user_text = build_targeted_prompt(row["prediction"], assert_text)
                chat_prompt = qwen_chat_prompt(tokenizer, targeted_user_text)
                prompt_token_len = len(tokenizer(chat_prompt, add_special_tokens=False).input_ids)

                plan.append({
                    "prompt": chat_prompt,
                    "prompt_token_len": int(prompt_token_len),

                    "task_id": tid,
                    "question_id": qid,
                    "buggy_code": str(row["prediction"] or ""),

                    "test_code": str(row["test_code"]),
                    "entry_point": row.get("entry_point", None),
                    "prompt_text": row.get("query", "") or "",
                    "target_assert_index": int(aidx),
                    "target_assert_text": assert_text,

                    "baseline_total": int(base["baseline_total"]),
                    "baseline_failed": int(base["baseline_failed"]),
                    "baseline_passed": int(base["baseline_passed"]),
                    "baseline_failing_indices": list(map(int, base["failing_indices"])),

                    "gold_completion": str(row.get("gold_completion", "") or ""),
                })

    rng.shuffle(plan)
    split = int(math.ceil((1.0 - EVAL_FRACTION) * len(plan)))
    train_rows, eval_rows = plan[:split], plan[split:]

    with open(CURRICULUM_JSONL, "w", encoding="utf-8") as f:
        for row in plan:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "total_unitcases": len(plan),
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "problems_covered": len({(r["task_id"], r["question_id"]) for r in plan}),
        "avg_unitcases_per_problem": float(len(plan)) / max(1, len({(r["task_id"], r["question_id"]) for r in plan})),
        "target_unitcases": TARGET_UNITCASE_SAMPLES,
        "max_unitcases_per_problem": MAX_UNITCASES_PER_PROBLEM,
        "input_tsv": INPUT_TSV,
        "qwen3_thinking": False,
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "min_p": MIN_P, "presence_penalty": PRESENCE_PENALTY},
    }
    with open(CURRICULUM_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if is_main_process():
        logging.info(f"Built plan: {len(plan)} unitcases | train={len(train_rows)} eval={len(eval_rows)}")
        logging.info(f"Stats saved to {CURRICULUM_STATS}; plan to {CURRICULUM_JSONL}")

    return train_rows, eval_rows

def rows_to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame(rows))

# =========================
# REWARD FUNCTION (TARGETED) — updated extraction
# =========================
def bugfix_reward(
    prompts: List[str],
    completions,
    completions_ids=None,
    task_id: List[str] = None,
    question_id: List[str] = None,
    buggy_code: List[str] = None,
    baseline_failed: List[int] = None,
    baseline_total: List[int] = None,
    baseline_passed: List[int] = None,
    baseline_failing_indices: List[List[int]] = None,
    test_code: List[str] = None,
    entry_point: List[Any] = None,
    prompt_text: List[str] = None,
    target_assert_index: List[int] = None,
    target_assert_text: List[str] = None,
    prompt_token_len: List[int] = None,  # in dataset (logging/analysis)

    gold_completion: List[str] = None,

    trainer_state=None,
    **kwargs
) -> List[float]:

    # Normalize completions to strings
    if completions and isinstance(completions[0], list):
        norm_completions = []
        for c in completions:
            if c and isinstance(c[0], dict):
                norm_completions.append(c[0].get("content", ""))
            else:
                norm_completions.append(str(c))
    else:
        norm_completions = [str(c) for c in completions]

    m = len(prompts) if prompts is not None else 0
    n = len(norm_completions)
    if m <= 0:
        return [0.0] * n

    k = max(1, n // m)

    def prompt_index(i: int) -> int:
        return min(m - 1, i // k)

    rewards: List[float] = []
    epoch_val = CURRENT_EPOCH
    global_step = None
    if isinstance(trainer_state, dict):
        global_step = trainer_state.get("global_step", None)

    batch_baseline_failed: List[int] = []

    # per-call aggregation lists
    pass_rate_before_list: List[float] = []
    pass_rate_after_list: List[float] = []
    delta_pass_rate_list: List[float] = []
    frac_improve_list: List[float] = []
    frac_regress_list: List[float] = []
    additional_fixed_list: List[int] = []
    additional_failed_list: List[int] = []
    reward_suite_signed_list: List[float] = []

    reward_base_list: List[float] = []
    reward_targeted_bonus_list: List[float] = []
    reward_suite_extra_list: List[float] = []

    lev_bug_fix_norm_list: List[float] = []
    ast_bug_fix_norm_list: List[float] = []

    rollout_counter = defaultdict(int)

    for i, comp in enumerate(norm_completions):
        j = prompt_index(i)

        ex_task_id     = task_id[j] if task_id else ""
        ex_question_id = question_id[j] if question_id else ""
        base_failed    = int(baseline_failed[j]) if baseline_failed else 0
        base_total     = int(baseline_total[j]) if baseline_total else 0
        base_passed    = int(baseline_passed[j]) if baseline_passed else max(0, base_total - base_failed)

        ex_test_code   = (test_code[j] if test_code else "") or ""
        ex_entry_point = entry_point[j] if entry_point else None
        ex_prompt_text = prompt_text[j] if prompt_text else ""
        tgt_idx        = int(target_assert_index[j]) if target_assert_index else -1
        tgt_text       = target_assert_text[j] if target_assert_text else f"assert #{tgt_idx}"

        ex_buggy_code  = str(buggy_code[j] if buggy_code is not None else "")
        ex_gold_code   = str(gold_completion[j] if gold_completion is not None else "")

        batch_baseline_failed.append(base_failed)

        key = (ex_task_id, ex_question_id, tgt_idx)
        r = rollout_counter[key]
        rollout_counter[key] += 1
        attempt_id = f"{ex_task_id}:{ex_question_id}:{tgt_idx}:e{epoch_val}:g{global_step}:r{r}"

        denom_base_total = max(1, base_total)
        pass_rate_before = base_passed / float(denom_base_total)

        # defaults
        total_after = base_total
        failed_after = base_failed
        passed_after = base_passed

        additional_fixed = 0
        additional_failed = 0
        frac_improve = 0.0
        frac_regress = 0.0
        reward_suite_signed = 0.0
        pass_rate_after = pass_rate_before
        delta_pass_rate = 0.0

        reward_base = 0.0
        targeted_bonus = 0.0
        suite_extra = 0.0
        reward = 0.0
        target_fixed = False

        # ===== NEW extraction pipeline =====
        gen_only = extract_generated_only_text(
            tokenizer=kwargs.get("tokenizer", None) or kwargs.get("processing_class", None) or None,
            prompt_text=prompts[j],
            completion_text=comp,
            completion_ids=(completions_ids[i] if completions_ids is not None else None),
            max_prompt_tokens=MAX_PROMPT_TOKENS,
        )

        # If kwargs didn't include tokenizer (some TRL versions), fall back safely:
        if gen_only is None or gen_only == "":
            # use text-only stripping if no tokenizer
            gen_only = _strip_think_blocks(_strip_prompt_echo_textwise(comp, prompts[j]))

        fixed_code = extract_last_python_fence(gen_only)

        # logging distances
        d_bug_fix = _pair_dists(ex_buggy_code, fixed_code)
        lev_bug_fix_norm_list.append(d_bug_fix["lev_norm"])
        ast_bug_fix_norm_list.append(d_bug_fix["ast_norm"])

        # invalid shape => reward 0
        if not fixed_code or not looks_like_function_or_class(fixed_code):
            reward = 0.0
            rewards.append(reward)

            pass_rate_before_list.append(pass_rate_before)
            pass_rate_after_list.append(pass_rate_after)
            delta_pass_rate_list.append(delta_pass_rate)
            additional_fixed_list.append(additional_fixed)
            additional_failed_list.append(additional_failed)
            frac_improve_list.append(frac_improve)
            frac_regress_list.append(frac_regress)
            reward_suite_signed_list.append(reward_suite_signed)

            reward_base_list.append(reward_base)
            reward_targeted_bonus_list.append(targeted_bonus)
            reward_suite_extra_list.append(suite_extra)

            append_jsonl({
                "event": "target_attempt",
                "attempt_id": attempt_id,
                "task_id": ex_task_id,
                "question_id": ex_question_id,
                "target_assert_index": tgt_idx,
                "target_assert_text_head": (tgt_text or "")[:500],

                "baseline_total": base_total,
                "baseline_failed": base_failed,
                "baseline_passed": base_passed,

                "after_total": total_after,
                "after_failed": failed_after,
                "after_passed": passed_after,

                "target_fixed": bool(target_fixed),

                "reward": float(reward),
                "reward_base": float(reward_base),
                "reward_targeted_bonus": float(targeted_bonus),
                "reward_suite_extra": float(suite_extra),
                "reward_suite_signed": float(reward_suite_signed),

                "additional_fixed": int(additional_fixed),
                "additional_failed": int(additional_failed),
                "frac_improve": float(frac_improve),
                "frac_regress": float(frac_regress),
                "pass_rate_before": float(pass_rate_before),
                "pass_rate_after": float(pass_rate_after),
                "delta_pass_rate": float(delta_pass_rate),

                "epoch": epoch_val,
                "global_step": global_step,
                "gen_index": i,
                "prompt_index": j,
                "rollout_index": r,
                "num_generations": NUM_GENERATIONS,

                "edit/bug_fix/lev": d_bug_fix["lev"],
                "edit/bug_fix/lev_norm": d_bug_fix["lev_norm"],
                "edit/bug_fix/ast": d_bug_fix["ast"],
                "edit/bug_fix/ast_norm": d_bug_fix["ast_norm"],

                "gold_code_present": bool(ex_gold_code and len(ex_gold_code) > 0),

                "prompt_token_len": int(prompt_token_len[j]) if prompt_token_len else None,
                "prompt_text_full": prompts[j],
                "generated_text_full": comp,
                "generated_only_text_full": gen_only,
                "fixed_code_full": fixed_code,
                "buggy_code_full": ex_buggy_code,
                "gold_code_full": ex_gold_code,
            })
            continue

        # 1) targeted assert
        try:
            single = eval_single_assert_with_output(ex_prompt_text, fixed_code, ex_test_code, ex_entry_point, tgt_idx)
            target_fixed = bool(single.get("ok", False))
        except Exception:
            target_fixed = False

        # 2) full suite
        try:
            full = eval_all_asserts_with_output(ex_prompt_text, fixed_code, ex_test_code, ex_entry_point)
        except Exception:
            full = {"results": [], "total": base_total}

        after_results = full.get("results", [])
        failed_after = len([rr for rr in after_results if not rr.get("ok")])
        total_after  = int(full.get("total", base_total))
        if total_after <= 0:
            total_after = base_total
        passed_after = max(0, total_after - failed_after)

        pass_rate_after = passed_after / float(max(1, total_after))
        delta_pass_rate = pass_rate_after - pass_rate_before

        additional_fixed = max(0, base_failed - failed_after)
        additional_failed = max(0, failed_after - base_failed)

        denom_improve = float(max(1, base_failed))
        denom_regress = float(max(1, base_passed))

        if failed_after <= base_failed:
            frac_improve = additional_fixed / denom_improve
            frac_regress = 0.0
            reward_suite_signed = 1000.0 * frac_improve
        else:
            frac_improve = 0.0
            frac_regress = additional_failed / denom_regress
            reward_suite_signed = -1000.0 * frac_regress

        # ===== PASS-RATE ONLY REWARD =====
        # Pure function of full-suite pass-rate improvement.
        # Range ~ [-1000, +1000] since delta_pass_rate in [-1, 1].
        targeted_bonus = 0.0
        suite_extra = 0.0
        # keep target_fixed for logging (already computed above), but it does NOT affect reward
        reward = 1000.0 * float(delta_pass_rate)
        reward_base = reward

        rewards.append(reward)

        pass_rate_before_list.append(pass_rate_before)
        pass_rate_after_list.append(pass_rate_after)
        delta_pass_rate_list.append(delta_pass_rate)
        additional_fixed_list.append(additional_fixed)
        additional_failed_list.append(additional_failed)
        frac_improve_list.append(frac_improve)
        frac_regress_list.append(frac_regress)
        reward_suite_signed_list.append(reward_suite_signed)

        reward_base_list.append(reward_base)
        reward_targeted_bonus_list.append(targeted_bonus)
        reward_suite_extra_list.append(suite_extra)

        append_jsonl({
            "event": "target_attempt",
            "attempt_id": attempt_id,
            "task_id": ex_task_id,
            "question_id": ex_question_id,
            "target_assert_index": tgt_idx,
            "target_assert_text_head": (tgt_text or "")[:500],

            "baseline_total": base_total,
            "baseline_failed": base_failed,
            "baseline_passed": base_passed,

            "after_total": total_after,
            "after_failed": failed_after,
            "after_passed": passed_after,

            "target_fixed": bool(target_fixed),

            "reward": float(reward),
            "reward_base": float(reward_base),
            "reward_targeted_bonus": float(targeted_bonus),
            "reward_suite_extra": float(suite_extra),
            "reward_suite_signed": float(reward_suite_signed),

            "additional_fixed": int(additional_fixed),
            "additional_failed": int(additional_failed),
            "frac_improve": float(frac_improve),
            "frac_regress": float(frac_regress),
            "pass_rate_before": float(pass_rate_before),
            "pass_rate_after": float(pass_rate_after),
            "delta_pass_rate": float(delta_pass_rate),

            "epoch": epoch_val,
            "global_step": global_step,
            "gen_index": i,
            "prompt_index": j,
            "rollout_index": r,
            "num_generations": NUM_GENERATIONS,

            "edit/bug_fix/lev": d_bug_fix["lev"],
            "edit/bug_fix/lev_norm": d_bug_fix["lev_norm"],
            "edit/bug_fix/ast": d_bug_fix["ast"],
            "edit/bug_fix/ast_norm": d_bug_fix["ast_norm"],

            "gold_code_present": bool(ex_gold_code and len(ex_gold_code) > 0),

            "prompt_token_len": int(prompt_token_len[j]) if prompt_token_len else None,
            "prompt_text_full": prompts[j],
            "generated_text_full": comp,
            "generated_only_text_full": gen_only,
            "fixed_code_full": fixed_code,
            "buggy_code_full": ex_buggy_code,
            "gold_code_full": ex_gold_code,
        })

    # rank-0 step logging + epoch buffers (unchanged)
    if is_main_process():
        arr = np.array(rewards, dtype=np.float32) if rewards else np.array([0.0], dtype=np.float32)
        base = np.array(batch_baseline_failed, dtype=np.float32) if batch_baseline_failed else np.array([0.0], dtype=np.float32)

        pr_before_arr = np.array(pass_rate_before_list, dtype=np.float32) if pass_rate_before_list else np.array([0.0], dtype=np.float32)
        pr_after_arr  = np.array(pass_rate_after_list, dtype=np.float32) if pass_rate_after_list else np.array([0.0], dtype=np.float32)
        dpr_arr       = np.array(delta_pass_rate_list, dtype=np.float32) if delta_pass_rate_list else np.array([0.0], dtype=np.float32)
        fi_arr        = np.array(frac_improve_list, dtype=np.float32) if frac_improve_list else np.array([0.0], dtype=np.float32)
        fr_arr        = np.array(frac_regress_list, dtype=np.float32) if frac_regress_list else np.array([0.0], dtype=np.float32)
        add_fixed_arr = np.array(additional_fixed_list, dtype=np.float32) if additional_fixed_list else np.array([0.0], dtype=np.float32)
        add_fail_arr  = np.array(additional_failed_list, dtype=np.float32) if additional_failed_list else np.array([0.0], dtype=np.float32)
        rsuite_arr    = np.array(reward_suite_signed_list, dtype=np.float32) if reward_suite_signed_list else np.array([0.0], dtype=np.float32)

        rbase_arr     = np.array(reward_base_list, dtype=np.float32) if reward_base_list else np.array([0.0], dtype=np.float32)
        rtb_arr       = np.array(reward_targeted_bonus_list, dtype=np.float32) if reward_targeted_bonus_list else np.array([0.0], dtype=np.float32)
        rse_arr       = np.array(reward_suite_extra_list, dtype=np.float32) if reward_suite_extra_list else np.array([0.0], dtype=np.float32)

        lbf_arr = np.array(lev_bug_fix_norm_list, dtype=np.float32) if lev_bug_fix_norm_list else np.array([0.0], dtype=np.float32)
        abf_arr = np.array(ast_bug_fix_norm_list, dtype=np.float32) if ast_bug_fix_norm_list else np.array([0.0], dtype=np.float32)

        try:
            wandb.log({
                "reward/mean": float(arr.mean()),
                "reward/std": float(arr.std()),
                "reward/max": float(arr.max()),
                "reward/p95": float(np.percentile(arr, 95)),
                "reward/fixed_rate": float((arr > 0).mean()),
                "reward/baseline_failed_mean": float(base.mean()),
                "reward/baseline_failed_std": float(base.std()),

                "reward/base_mean": float(rbase_arr.mean()),
                "reward/targeted_bonus_mean": float(rtb_arr.mean()),
                "reward/suite_extra_mean": float(rse_arr.mean()),

                "suite/reward_signed_mean": float(rsuite_arr.mean()),
                "suite/frac_improve_mean": float(fi_arr.mean()),
                "suite/frac_regress_mean": float(fr_arr.mean()),
                "suite/additional_fixed_mean": float(add_fixed_arr.mean()),
                "suite/additional_failed_mean": float(add_fail_arr.mean()),
                "suite/pass_rate_before_mean": float(pr_before_arr.mean()),
                "suite/pass_rate_after_mean": float(pr_after_arr.mean()),
                "suite/delta_pass_rate_mean": float(dpr_arr.mean()),

                "edit/lev_bug_fix_norm_mean": float(lbf_arr.mean()),
                "edit/ast_bug_fix_norm_mean": float(abf_arr.mean()),

                "epoch": CURRENT_EPOCH,
                "trainer/global_step": global_step,
            })
            try:
                wandb.log({
                    "hist/pass_rate_after": wandb.Histogram(pr_after_arr, num_bins=20),
                    "hist/delta_pass_rate": wandb.Histogram(dpr_arr, num_bins=20),
                    "hist/reward": wandb.Histogram(arr, num_bins=20),
                    "hist/reward_base": wandb.Histogram(rbase_arr, num_bins=20),
                })
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] wandb logging failed in reward fn: {e}")

        EPOCH_BUF["rewards"].extend(arr.tolist())
        EPOCH_BUF["fixed_flags"].extend((arr > 0).astype(np.int8).tolist())
        EPOCH_BUF["baseline_failed"].extend(base.tolist())

        EPOCH_BUF["reward_base"].extend(rbase_arr.tolist())
        EPOCH_BUF["reward_targeted_bonus"].extend(rtb_arr.tolist())
        EPOCH_BUF["reward_suite_extra"].extend(rse_arr.tolist())

        EPOCH_BUF["pass_rate_before"].extend(pr_before_arr.tolist())
        EPOCH_BUF["pass_rate_after"].extend(pr_after_arr.tolist())
        EPOCH_BUF["delta_pass_rate"].extend(dpr_arr.tolist())
        EPOCH_BUF["frac_improve"].extend(fi_arr.tolist())
        EPOCH_BUF["frac_regress"].extend(fr_arr.tolist())
        EPOCH_BUF["additional_fixed"].extend(add_fixed_arr.tolist())
        EPOCH_BUF["additional_failed"].extend(add_fail_arr.tolist())
        EPOCH_BUF["reward_suite_signed"].extend(rsuite_arr.tolist())

        EPOCH_BUF["lev_bug_fix_norm"].extend(lbf_arr.tolist())
        EPOCH_BUF["ast_bug_fix_norm"].extend(abf_arr.tolist())

    return rewards

# =========================
# HELD-OUT EVAL (sampling; avoids greedy in thinking mode)
# =========================
@torch.no_grad()
def run_heldout_eval(model, tokenizer, rows: List[Dict[str, Any]], max_items: int = 64) -> Dict[str, float]:
    model_ = model.module if hasattr(model, "module") else model
    model_.eval()

    take = rows[:max_items] if max_items and len(rows) > max_items else rows
    rewards, success_flags = [], []

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    for ex in take:
        prompt = ex["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model_.parameters()).device)

        gen = model_.generate(
            **inputs,
            max_new_tokens=MAX_COMPLETION_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        try:
            gen_only_ids = gen[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        except Exception:
            text = tokenizer.decode(gen[0], skip_special_tokens=True)

        text = _strip_think_blocks(text)
        fixed = extract_last_python_fence(text)

        try:
            single = eval_single_assert_with_output(
                ex.get("prompt_text", "") or "",
                fixed,
                ex["test_code"],
                ex["entry_point"],
                int(ex["target_assert_index"])
            )
            target_fixed = bool(single.get("ok", False))
        except Exception:
            target_fixed = False

        try:
            full = eval_all_asserts_with_output(
                ex.get("prompt_text", "") or "",
                fixed,
                ex["test_code"],
                ex["entry_point"]
            )
        except Exception:
            full = {"results": [], "total": ex.get("baseline_total", 0)}

        after_results = full.get("results", [])
        failed_after = len([r for r in after_results if not r.get("ok")])
        base_failed = int(ex.get("baseline_failed", 0))

        base_total = int(ex.get("baseline_total", 0))
        base_passed = int(ex.get("baseline_passed", max(0, base_total - base_failed)))
        pass_rate_before = base_passed / float(max(1, base_total))

        total_after = int(full.get("total", base_total))
        if total_after <= 0:
            total_after = base_total
        passed_after = max(0, total_after - failed_after)
        pass_rate_after = passed_after / float(max(1, total_after))

        reward = 1000.0 * float(pass_rate_after - pass_rate_before)

        rewards.append(reward)
        success_flags.append(1.0 if reward > 0 else 0.0)

    arr = np.array(rewards, dtype=np.float32) if rewards else np.array([0.0], dtype=np.float32)
    succ = np.array(success_flags, dtype=np.float32) if success_flags else np.array([0.0], dtype=np.float32)
    return {
        "eval/reward_mean": float(arr.mean()),
        "eval/reward_p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "eval/fixed_rate": float(succ.mean()),
        "eval/n": float(len(arr)),
    }

# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Qwen3-8B PATSSEL debugger with the paper's GRPO objective.")
    p.add_argument("--train_file", type=str, default=INPUT_TSV)
    p.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    p.add_argument("--tester_path", type=str, default=TESTER_PATH)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def apply_runtime_args(args: argparse.Namespace) -> None:
    global INPUT_TSV, OUTPUT_DIR, MODEL_ID, TESTER_PATH, SEED
    global CURRICULUM_JSONL, CURRICULUM_STATS, LOG_JSONL, WANDB_RUN_ID_FILE
    INPUT_TSV = args.train_file
    OUTPUT_DIR = args.output_dir
    MODEL_ID = args.model_id
    TESTER_PATH = args.tester_path
    SEED = args.seed
    CURRICULUM_JSONL = os.path.join(OUTPUT_DIR, "training_unitcases_plan.jsonl")
    CURRICULUM_STATS = os.path.join(OUTPUT_DIR, "training_unitcases_stats.json")
    LOG_JSONL = os.path.join(OUTPUT_DIR, "training_curriculum_targeted_logs_train_Qwen3_8B_random_split.jsonl")
    WANDB_RUN_ID_FILE = os.path.join(OUTPUT_DIR, "wandb_run_id.txt")

# =========================
# MAIN
# =========================
def main():
    args_cli = parse_args()
    apply_runtime_args(args_cli)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global _tester, evaluate_code, eval_with_timeout, eval_all_asserts_with_output, eval_single_assert_with_output, extract_assert_texts, resolve_entry_point
    _tester = import_tester(TESTER_PATH)
    evaluate_code = _tester.evaluate_code
    eval_with_timeout = _tester.eval_with_timeout
    eval_all_asserts_with_output = _tester.eval_all_asserts_with_output
    eval_single_assert_with_output = _tester.eval_single_assert_with_output
    extract_assert_texts = _tester.extract_assert_texts
    resolve_entry_point = _tester.resolve_entry_point

    append_jsonl({
        "event": "script_invocation",
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "model_id": MODEL_ID,
        "input_tsv": INPUT_TSV,
        "output_dir": OUTPUT_DIR,
        "qwen3_thinking": False,
    })

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_rows, eval_rows = build_training_plan(tokenizer)
    train_ds = rows_to_dataset(train_rows)

    global EVAL_ROWS
    EVAL_ROWS = eval_rows

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
        device_map=None,
    )
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        max_grad_norm=CLIP_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,

        save_safetensors=True,

        max_prompt_length=MAX_PROMPT_TOKENS,
        num_generations=NUM_GENERATIONS,
        # generation_batch_size=8,
        generation_batch_size=4,
        max_completion_length=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,

        logging_steps=LOG_EVERY,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        log_completions=False,

        loss_type="grpo",
        beta=BETA,
        scale_rewards="batch",

        fp16=True,
        bf16=False,
        remove_unused_columns=False,

        save_strategy="steps",
        save_steps=SAVE_EVERY_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        ddp_find_unused_parameters=False,
        report_to=["wandb"],
    )

    # Best-effort: pass top_k/min_p/presence_penalty if supported
    for name, value in [("top_k", TOP_K), ("min_p", MIN_P), ("presence_penalty", PRESENCE_PENALTY)]:
        try:
            if hasattr(args, name):
                setattr(args, name, value)
        except Exception:
            pass

    if is_main_process():
        prev_run_id = None
        if os.path.exists(WANDB_RUN_ID_FILE):
            with open(WANDB_RUN_ID_FILE) as f:
                prev_run_id = f.read().strip() or None

        wandb.init(
            project="BugFixer-GRPO",
            name="8B_only_passrate_from_random_split_1200_fixed_mapping_cot",
            id=prev_run_id,
            resume="allow",
            config={
                "script_path": SCRIPT_PATH,
                "model": MODEL_ID,
                "epochs": EPOCHS,
                "num_generations": NUM_GENERATIONS,
                "batch_size_groups": BATCH_SIZE,
                "lr": LR,
                "beta": BETA,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "min_p": MIN_P,
                "presence_penalty": PRESENCE_PENALTY,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "clip_grad_norm": CLIP_GRAD_NORM,
                "dataset": "8B_only_passrate_from_random_split_1200_fixed_mapping_cot",
                "target_unitcases": TARGET_UNITCASE_SAMPLES,
                "max_unitcases_per_problem": MAX_UNITCASES_PER_PROBLEM,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_targets": ",".join(LORA_TARGET_MODULES),
                "qwen3_thinking": False,
            }
        )

        try:
            if prev_run_id is None and wandb.run and wandb.run.id:
                with open(WANDB_RUN_ID_FILE, "w") as f:
                    f.write(wandb.run.id)
        except Exception as e:
            logging.warning(f"Could not persist W&B run id: {e}")

    # NOTE: We pass tokenizer to reward fn via kwargs by binding it in a closure
    def bugfix_reward_with_tok(*a, **kw):
        kw["tokenizer"] = tokenizer
        return bugfix_reward(*a, **kw)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=bugfix_reward_with_tok,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[
            EpochTrackerCallback(),
            WandbTrainingCallback(),
            (lambda tok: type(
                "CB", (EpochEndLoggerAndEvalCallback,),
                {"on_epoch_end": lambda self, *a, **kw: EpochEndLoggerAndEvalCallback.on_epoch_end(self, *a, tokenizer=tok, **kw)}
            ))(tokenizer)()
        ],
        peft_config=lora_cfg,
    )

    logging.info(f"Train rows: {len(train_ds)} | per-device batch={BATCH_SIZE} | k={NUM_GENERATIONS}")
    append_jsonl({
        "event": "training_start",
        "epochs": EPOCHS,
        "num_generations": NUM_GENERATIONS,
        "batch_size_groups": BATCH_SIZE,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "min_p": MIN_P,
        "presence_penalty": PRESENCE_PENALTY,
        "input_tsv": INPUT_TSV,
        "qwen3_thinking": False,
    })

    resume_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        logging.info(f"[resume] Resuming from checkpoint: {resume_ckpt}")
        append_jsonl({"event": "resume_from_checkpoint", "path": resume_ckpt})
    else:
        logging.info("[resume] No checkpoint found. Starting fresh.")
        append_jsonl({"event": "resume_from_checkpoint", "path": None})

    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    trainer.save_model(OUTPUT_DIR)
    if trainer.processing_class and hasattr(trainer.processing_class, "save_pretrained"):
        trainer.processing_class.save_pretrained(OUTPUT_DIR)

    logging.info(f"Saved final checkpoint to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()