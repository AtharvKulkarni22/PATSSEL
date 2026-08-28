#!/usr/bin/env python3
"""
GRPO training for a Qwen 4B unit-test generator.

Stage-1 reward:
    - invalid format/check function: -200
    - zero asserts: -200
    - fewer than MIN_ASSERTS_REQUIRED asserts: 0
    - otherwise: 1000 * gold_pass_rate + 20 * min(correct_asserts, 10)

This trains the model to generate many valid deterministic tests whose expected outputs
are correct with respect to the provided gold solution.
"""
from __future__ import annotations

import os
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import sys, re, ast, json, math, random, logging, textwrap, argparse, importlib.util
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

try:
    from trl import GRPOTrainer, GRPOConfig
except Exception:
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
        from trl.trainer.grpo_config import GRPOConfig
    except Exception as e:
        raise ImportError("GRPOTrainer not found. Please upgrade TRL >= 0.9.x. Original error: " + str(e))

from peft import LoraConfig

SCRIPT_PATH = os.path.abspath(__file__)
LOG_CONTEXT = {"script_path": SCRIPT_PATH}

CACHE_ROOT = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
TRANSFORMERS_CACHE = os.environ.get("TRANSFORMERS_CACHE", os.path.join(CACHE_ROOT, "transformers"))
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", os.path.join(CACHE_ROOT, "datasets"))
CACHE_DIR = TRANSFORMERS_CACHE

# Defaults matching your existing scaffold.
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_NAME = "newfacade/LeetCodeDataset"
DATASET_SPLIT = "train"
OUTPUT_DIR = str(_REPO_ROOT / "checkpoints/test_generator")
TESTER_PATH = str(_REPO_ROOT / "patssel/evaluation/leetcode.py")
EXCLUDE_TASK_IDS_JSON = str(_REPO_ROOT / "data/leetcode/eval_task_ids.json")

CURRICULUM_JSONL = os.path.join(OUTPUT_DIR, "training_testgen_gold_plan.jsonl")
CURRICULUM_STATS = os.path.join(OUTPUT_DIR, "training_testgen_gold_stats.json")
LOG_JSONL = os.path.join(OUTPUT_DIR, "training_testgen_gold_attempt_logs.jsonl")
WANDB_RUN_ID_FILE = os.path.join(OUTPUT_DIR, "wandb_run_id.txt")

SEED = 1012
EVAL_FRACTION = 0.02
MAX_TRAIN_PROBLEMS: Optional[int] = None
NUM_FEWSHOT = 3

MIN_ASSERTS_REQUIRED = 10
INVALID_REWARD = -200.0
SHORT_SUITE_REWARD = 0.0
GOLD_PASS_SCALE = 1000.0
CORRECT_ASSERT_BONUS = 20.0
CORRECT_ASSERT_BONUS_CAP = 10

NUM_GENERATIONS = 4
BATCH_SIZE = 1
EPOCHS = 1
MAX_PROMPT_TOKENS = 3072
MAX_COMPLETION_TOKENS = 1800
GENERATION_BATCH_SIZE = 4

TEMPERATURE = 0.7
TOP_P = 0.9
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
SAVE_EVERY_STEPS = 100
SAVE_TOTAL_LIMIT = 50

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def append_jsonl(obj: Dict[str, Any]) -> None:
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(LOG_JSONL), exist_ok=True)
    payload = {**LOG_CONTEXT, **obj}
    with open(LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def import_tester(path: str):
    spec = importlib.util.spec_from_file_location("tester", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import tester module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    required = ["resolve_entry_point", "eval_all_asserts_with_output", "eval_single_assert_with_output", "extract_assert_texts", "count_asserts", "TIME_LIMIT_SEC"]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"tester.py must define {name}")
    return mod


_tester = None
resolve_entry_point = None
eval_all_asserts_with_output = None
eval_single_assert_with_output = None
extract_assert_texts = None
count_asserts = None

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
- Generate AT LEAST 10 assert statements.
- Generate diverse tests that aim for broad behavior coverage of the provided correct solution.
- Include easy, edge, boundary, duplicate, large-value, adversarial, empty/singleton, and structure-specific cases when applicable.
- Every assert must be deterministic.
- Every expected output must be correct with respect to the provided correct solution.
- If no valid output exists for an input, you may use `None` only when consistent with the provided correct solution behavior.
"""

PY_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
CHECK_HEADER_RE = re.compile(r"^\s*def\s+check\s*\(\s*candidate\s*\)\s*:\s*$")
ASSERT_START_RE = re.compile(r"^\s*assert\b")
SAFE_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*$")
DISALLOWED_HELPER_PREFIXES = ("import ", "from ", "def ", "class ", "for ", "while ", "if ", "elif ", "else:", "try:", "except", "finally:", "with ", "return ", "yield ", "@")


def normalize_whitespace(text: str) -> str:
    return textwrap.dedent(text or "").strip()


def _strip_think_blocks(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text or "").strip()


def _strip_prompt_echo_textwise(text: str, prompt: str) -> str:
    if not text:
        return ""
    if prompt and text.startswith(prompt):
        return text[len(prompt):]
    return text


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


def extract_generated_only_text(tokenizer, prompt_text: str, completion_text: str, completion_ids: Optional[Any] = None, *, max_prompt_tokens: Optional[int] = None, end_think_token_id: Optional[int] = 151668) -> str:
    try:
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        if max_prompt_tokens is not None:
            prompt_len = min(prompt_len, int(max_prompt_tokens))
    except Exception:
        prompt_len = None
    ids: Optional[List[int]] = None
    if completion_ids is not None:
        try:
            ids = completion_ids.detach().cpu().tolist() if hasattr(completion_ids, "detach") else completion_ids
            if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
                ids = ids[0]
        except Exception:
            ids = None
    if isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids):
        gen_ids = ids[prompt_len:] if prompt_len is not None and len(ids) > prompt_len else ids
        if end_think_token_id is not None:
            try:
                if end_think_token_id in gen_ids:
                    idx = len(gen_ids) - gen_ids[::-1].index(end_think_token_id)
                    gen_ids = gen_ids[idx:]
            except Exception:
                pass
        try:
            return _strip_think_blocks(tokenizer.decode(gen_ids, skip_special_tokens=True).strip("\n"))
        except Exception:
            return _strip_think_blocks(completion_text or "")
    return _strip_think_blocks(_strip_prompt_echo_textwise(completion_text or "", prompt_text)).strip()


def _is_safe_expr(expr_src: str) -> bool:
    try:
        tree = ast.parse(expr_src, mode="eval")
    except Exception:
        return False
    allowed_call_names = {"list_node", "tree_node", "sorted", "tuple", "list", "set", "dict", "len", "sum", "min", "max", "abs", "range"}
    class Visitor(ast.NodeVisitor):
        ok = True
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_call_names:
                for arg in node.args: self.visit(arg)
                for kw in node.keywords: self.visit(kw.value)
                return
            self.ok = False
        def visit_Attribute(self, node: ast.Attribute): self.ok = False
        def visit_Lambda(self, node: ast.Lambda): self.ok = False
        def visit_ListComp(self, node: ast.ListComp): self.ok = False
        def visit_SetComp(self, node: ast.SetComp): self.ok = False
        def visit_DictComp(self, node: ast.DictComp): self.ok = False
        def visit_GeneratorExp(self, node: ast.GeneratorExp): self.ok = False
        def generic_visit(self, node):
            allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.UnaryOp, ast.UAdd, ast.USub, ast.Not, ast.Invert, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Subscript, ast.Slice, ast.Index, ast.Call)
            if not isinstance(node, allowed):
                self.ok = False
                return
            super().generic_visit(node)
    v = Visitor(); v.visit(tree)
    return bool(v.ok)


def _parse_single_stmt(stmt_src: str) -> Optional[ast.stmt]:
    try:
        tree = ast.parse(stmt_src)
        return tree.body[0] if len(tree.body) == 1 else None
    except Exception:
        return None


def _is_safe_helper_stmt(stmt_src: str) -> bool:
    stripped = stmt_src.strip()
    if not stripped: return False
    low = stripped.lower()
    if any(low.startswith(prefix) for prefix in DISALLOWED_HELPER_PREFIXES): return False
    if CHECK_HEADER_RE.match(stripped) or stripped.startswith("#"): return False
    stmt = _parse_single_stmt(stripped)
    if stmt is None or not isinstance(stmt, ast.Assign): return False
    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name): return False
    m = SAFE_ASSIGN_RE.match(stripped)
    return bool(m and _is_safe_expr(m.group(2)))


def _count_balance(text: str) -> int:
    score = 0; pairs = {"(": ")", "[": "]", "{": "}"}; closing = {")", "]", "}"}
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


def salvage_structured_blocks(text: str) -> Dict[str, Any]:
    src = strip_markdown_fences(text)
    lines = src.splitlines()
    helpers: List[str] = []; asserts: List[str] = []; dropped: List[str] = []
    seen_helpers: Set[str] = set(); seen_asserts: Set[str] = set()
    i = 0
    while i < len(lines):
        stripped = lines[i].rstrip("\n").strip()
        if not stripped or stripped.startswith("#") or CHECK_HEADER_RE.match(stripped) or stripped.startswith("```"):
            i += 1; continue
        if ASSERT_START_RE.match(stripped):
            block_lines = [stripped]; j = i + 1
            while True:
                joined = "\n".join(block_lines)
                stmt = _parse_single_stmt(joined)
                if isinstance(stmt, ast.Assert):
                    if joined not in seen_asserts:
                        seen_asserts.add(joined); asserts.append(joined)
                    i = j; break
                if j >= len(lines):
                    dropped.append(joined); i = j; break
                nxt = lines[j].rstrip("\n").strip()
                if (ASSERT_START_RE.match(nxt) or CHECK_HEADER_RE.match(nxt)) and _count_balance(joined) <= 0:
                    dropped.append(joined); i = j; break
                if nxt.startswith("```"):
                    dropped.append(joined); i = j + 1; break
                block_lines.append(nxt); j += 1
            continue
        if _is_safe_helper_stmt(stripped):
            if stripped not in seen_helpers:
                seen_helpers.add(stripped); helpers.append(stripped)
            i += 1; continue
        if "  #" in stripped:
            head = stripped.split("  #", 1)[0].rstrip()
            if _is_safe_helper_stmt(head):
                if head not in seen_helpers:
                    seen_helpers.add(head); helpers.append(head)
                i += 1; continue
        dropped.append(stripped); i += 1
    clean_lines = ["def check(candidate):"]
    if not helpers and not asserts:
        clean_lines.append("    pass")
    else:
        for h in helpers: clean_lines.append(f"    {h}")
        for a in asserts: clean_lines.extend("    " + ln for ln in a.splitlines())
    clean_test_code = "\n".join(clean_lines)
    parse_error = ""
    try: ast.parse(clean_test_code)
    except Exception as e: parse_error = f"{type(e).__name__}: {e}"
    return {"clean_test_code": clean_test_code, "helper_lines": helpers, "assert_blocks": asserts, "num_helper_lines": len(helpers), "num_assert_blocks": len(asserts), "dropped_lines": dropped, "parse_error": parse_error}


def salvage_clean_test_code(raw_generation: str, extracted_test_code: str) -> Dict[str, Any]:
    first = salvage_structured_blocks(extracted_test_code)
    second = salvage_structured_blocks(raw_generation)
    merged_helpers: List[str] = []; merged_asserts: List[str] = []; seen_h: Set[str] = set(); seen_a: Set[str] = set()
    for item in first["helper_lines"] + second["helper_lines"]:
        if item not in seen_h:
            seen_h.add(item); merged_helpers.append(item)
    for item in first["assert_blocks"] + second["assert_blocks"]:
        if item not in seen_a:
            seen_a.add(item); merged_asserts.append(item)
    clean_lines = ["def check(candidate):"]
    if not merged_helpers and not merged_asserts:
        clean_lines.append("    pass")
    else:
        for h in merged_helpers: clean_lines.append(f"    {h}")
        for a in merged_asserts: clean_lines.extend("    " + ln for ln in a.splitlines())
    clean_test_code = "\n".join(clean_lines)
    parse_error = ""
    try: ast.parse(clean_test_code)
    except Exception as e: parse_error = f"{type(e).__name__}: {e}"
    return {"clean_test_code": clean_test_code, "salvaged_helper_lines": merged_helpers, "salvaged_assert_blocks": merged_asserts, "num_salvaged_helper_lines": len(merged_helpers), "num_salvaged_assert_blocks": len(merged_asserts), "dropped_lines": first["dropped_lines"] + second["dropped_lines"], "parse_error": parse_error, "salvage_mode": "full_parse" if not parse_error and looks_like_check_function(extracted_test_code) else "structured_salvage"}


def make_fewshot_user(example: Dict[str, Any]) -> str:
    return f"Problem description:\n{normalize_whitespace(example.get('problem_description') or example.get('query') or example.get('prompt') or '')}\n\nStarter code:\n{normalize_whitespace(example.get('starter_code') or '')}\n\nCorrect solution:\n{normalize_whitespace(example.get('completion') or '')}\n\nEntry point:\n{example.get('entry_point') or ''}\n\nGenerate Python tests in the dataset format. Generate at least {MIN_ASSERTS_REQUIRED} assert statements."


def make_fewshot_assistant(example: Dict[str, Any]) -> str:
    test_code = normalize_whitespace(example.get("test") or example.get("test_code") or "")
    return f"```python\n{test_code}\n```"


def get_problem_text(row: Dict[str, Any]) -> str:
    return row.get("problem_description") or row.get("query") or row.get("prompt") or ""


def build_user_prompt(row: Dict[str, Any]) -> str:
    return f"Problem description:\n{normalize_whitespace(row.get('problem_description') or row.get('query') or row.get('prompt') or '')}\n\nStarter code:\n{normalize_whitespace(row.get('starter_code') or '')}\n\nCorrect solution:\n{normalize_whitespace(row.get('completion') or '')}\n\nEntry point:\n{row.get('entry_point') or ''}\n\nGenerate Python tests in the dataset format now.\nYou MUST generate at least {MIN_ASSERTS_REQUIRED} assert statements inside def check(candidate).\nGenerate diverse tests; do not output one simple test case.\nAll expected outputs must be correct with respect to the provided correct solution."


def qwen_chat_prompt(tokenizer, row: Dict[str, Any], fewshot_examples: List[Dict[str, Any]]) -> str:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": make_fewshot_user(ex)})
        messages.append({"role": "assistant", "content": make_fewshot_assistant(ex)})
    messages.append({"role": "user", "content": build_user_prompt(row)})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_dataset_row(row: Dict[str, Any]) -> Dict[str, Any]:
    completion = normalize_whitespace(row.get("completion") or "")
    entry = row.get("entry_point") or ""
    try:
        resolved_entry = resolve_entry_point(entry, completion)
    except Exception:
        resolved_entry = entry or ""
    return {"task_id": str(row.get("task_id", "")), "question_id": str(row.get("question_id", "")), "problem_description": get_problem_text(row), "starter_code": row.get("starter_code") or "", "completion": completion, "entry_point": resolved_entry, "reference_test": row.get("test") or row.get("test_code") or "", "difficulty": row.get("difficulty", "")}


def load_excluded_task_ids(path: Optional[str]) -> Set[str]:
    """Load task IDs that must not appear in training or few-shot examples."""
    if path is None or not str(path).strip():
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f"exclude task-id file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"exclude task-id file must contain a JSON list: {path}")
    return {str(x).strip() for x in data if str(x).strip()}


def choose_fewshot_examples(rows: List[Dict[str, Any]], k: int, exclude_task_id: str) -> List[Dict[str, Any]]:
    examples = [r for r in rows if str(r.get("task_id", "")) != str(exclude_task_id) and (r.get("test") or r.get("test_code")) and r.get("completion")]
    return examples[:k]


def load_training_rows(tokenizer, dataset_name: str, split_name: str, max_train_problems: Optional[int], exclude_task_ids_json: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ds = load_dataset(dataset_name, cache_dir=HF_DATASETS_CACHE)
    if split_name not in ds:
        raise ValueError(f"Split {split_name!r} not in dataset. Available: {list(ds.keys())}")
    raw_rows_all = [dict(x) for x in ds[split_name]]
    excluded_task_ids = load_excluded_task_ids(exclude_task_ids_json)
    excluded_present = {str(r.get("task_id", "")).strip() for r in raw_rows_all if str(r.get("task_id", "")).strip() in excluded_task_ids}
    raw_rows = [r for r in raw_rows_all if str(r.get("task_id", "")).strip() not in excluded_task_ids]
    raw_rows = [r for r in raw_rows if (r.get("completion") is not None and str(r.get("completion")).strip()) and get_problem_text(r).strip() and (r.get("entry_point") is not None and str(r.get("entry_point")).strip())]
    if is_main_process():
        logging.info("Loaded %d rows from %s/%s", len(raw_rows_all), dataset_name, split_name)
        logging.info("Excluded %d task ids from %s; %d were present in this split", len(excluded_task_ids), exclude_task_ids_json, len(excluded_present))
        logging.info("Rows remaining after exclusion and basic filters: %d", len(raw_rows))
    rng = np.random.default_rng(SEED)
    idxs = np.arange(len(raw_rows)); rng.shuffle(idxs)
    raw_rows = [raw_rows[int(i)] for i in idxs]
    if max_train_problems is not None:
        raw_rows = raw_rows[: int(max_train_problems)]
    plan: List[Dict[str, Any]] = []
    for r in raw_rows:
        norm = normalize_dataset_row(r)
        fewshots = choose_fewshot_examples(raw_rows, NUM_FEWSHOT, norm["task_id"])
        prompt = qwen_chat_prompt(tokenizer, norm, fewshots)
        prompt_token_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if prompt_token_len > MAX_PROMPT_TOKENS:
            prompt = qwen_chat_prompt(tokenizer, norm, [])
            prompt_token_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if prompt_token_len > MAX_PROMPT_TOKENS:
            continue
        plan.append({"prompt": prompt, "prompt_token_len": int(prompt_token_len), "task_id": norm["task_id"], "question_id": norm["question_id"], "problem_description": norm["problem_description"], "starter_code": norm["starter_code"], "gold_completion": norm["completion"], "entry_point": norm["entry_point"], "reference_test": norm["reference_test"], "difficulty": norm["difficulty"], "fewshot_task_ids": [str(x.get("task_id", "")) for x in fewshots]})
    rng.shuffle(plan)
    split = int(math.ceil((1.0 - EVAL_FRACTION) * len(plan)))
    train_rows, eval_rows = plan[:split], plan[split:]
    if is_main_process():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(CURRICULUM_JSONL, "w", encoding="utf-8") as f:
            for row in plan: f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats = {"total_rows": len(plan), "train_count": len(train_rows), "eval_count": len(eval_rows), "dataset_name": dataset_name, "dataset_split": split_name, "model_id": MODEL_ID, "min_asserts_required": MIN_ASSERTS_REQUIRED, "num_fewshot": NUM_FEWSHOT, "max_prompt_tokens": MAX_PROMPT_TOKENS, "max_completion_tokens": MAX_COMPLETION_TOKENS, "exclude_task_ids_json": exclude_task_ids_json, "excluded_task_ids_count": len(excluded_task_ids), "excluded_present_in_split_count": len(excluded_present), "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K, "min_p": MIN_P, "presence_penalty": PRESENCE_PENALTY}}
        with open(CURRICULUM_STATS, "w", encoding="utf-8") as f: json.dump(stats, f, ensure_ascii=False, indent=2)
        logging.info("Built testgen plan: %d rows | train=%d eval=%d", len(plan), len(train_rows), len(eval_rows))
    return train_rows, eval_rows


def rows_to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


CURRENT_EPOCH = 0
EVAL_ROWS: List[Dict[str, Any]] = []
EPOCH_BUF: Dict[str, List[Any]] = {"reward": [], "valid_format": [], "wellformed": [], "num_asserts": [], "correct_asserts": [], "incorrect_asserts": [], "gold_pass_rate": [], "short_suite": [], "parse_error": []}


class EpochTrackerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        global CURRENT_EPOCH
        if state.epoch is not None:
            try: CURRENT_EPOCH = int(state.epoch)
            except Exception:
                try: CURRENT_EPOCH = int(round(float(state.epoch)))
                except Exception: pass


class WandbTrainingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not is_main_process() or not logs: return
        passthrough = {f"train/{k}": v for k, v in logs.items() if any(s in k.lower() for s in ["entropy", "logp", "logps", "kl", "loss", "learning_rate"])}
        if passthrough: wandb.log(passthrough)
    def on_step_end(self, args, state, control, **kwargs):
        if not is_main_process(): return
        payload = {"epoch": state.epoch, "step": state.global_step}
        if state.log_history:
            last = state.log_history[-1]
            for key in ("loss", "learning_rate"):
                if key in last: payload[key] = last[key]
        wandb.log(payload)


class EpochEndLoggerAndEvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if not is_main_process(): return
        if EPOCH_BUF["reward"]:
            def mean_float(k: str) -> float:
                arr = np.array(EPOCH_BUF[k], dtype=np.float32)
                return float(arr.mean()) if arr.size else 0.0
            reward_arr = np.array(EPOCH_BUF["reward"], dtype=np.float32)
            wandb.log({"epoch/reward_mean": float(reward_arr.mean()), "epoch/reward_p95": float(np.percentile(reward_arr, 95)), "epoch/valid_format_rate": mean_float("valid_format"), "epoch/wellformed_rate": mean_float("wellformed"), "epoch/num_asserts_mean": mean_float("num_asserts"), "epoch/correct_asserts_mean": mean_float("correct_asserts"), "epoch/incorrect_asserts_mean": mean_float("incorrect_asserts"), "epoch/gold_pass_rate_mean": mean_float("gold_pass_rate"), "epoch/short_suite_rate": mean_float("short_suite"), "epoch/parse_error_rate": mean_float("parse_error"), "epoch": CURRENT_EPOCH})
            try:
                wandb.log({"epoch/reward_hist": wandb.Histogram(reward_arr, num_bins=20), "epoch/num_asserts_hist": wandb.Histogram(np.array(EPOCH_BUF["num_asserts"], dtype=np.float32), num_bins=20)})
            except Exception: pass
        for k in EPOCH_BUF: EPOCH_BUF[k].clear()
        if tokenizer is not None and model is not None and EVAL_ROWS:
            eval_metrics = run_heldout_eval(model, tokenizer, EVAL_ROWS)
            eval_metrics["epoch"] = CURRENT_EPOCH
            wandb.log(eval_metrics)


def normalize_completions(completions) -> List[str]:
    if completions and isinstance(completions[0], list):
        out = []
        for c in completions:
            if c and isinstance(c[0], dict): out.append(c[0].get("content", ""))
            else: out.append(str(c))
        return out
    return [str(c) for c in completions]


def evaluate_generated_tests_on_gold(prompt_text: str, gold_code: str, test_code: str, entry_point: str) -> Dict[str, Any]:
    if not looks_like_check_function(test_code):
        return {"valid_format": False, "wellformed": False, "num_asserts": 0, "correct_asserts": 0, "incorrect_asserts": 0, "gold_pass_rate": 0.0, "error": "missing def check(candidate)", "results": []}
    try: n_asserts = int(count_asserts(test_code))
    except Exception: n_asserts = 0
    if n_asserts <= 0:
        return {"valid_format": True, "wellformed": False, "num_asserts": 0, "correct_asserts": 0, "incorrect_asserts": 0, "gold_pass_rate": 0.0, "error": "no asserts", "results": []}
    try:
        eval_res = eval_all_asserts_with_output(prompt_text, gold_code, test_code, entry_point)
        results = eval_res.get("results", [])
        total = max(int(eval_res.get("total", n_asserts)), n_asserts)
        correct = sum(1 for r in results if r.get("ok"))
        incorrect = max(0, total - correct)
        return {"valid_format": True, "wellformed": True, "num_asserts": total, "correct_asserts": correct, "incorrect_asserts": incorrect, "gold_pass_rate": correct / float(max(1, total)), "error": eval_res.get("error", ""), "results": results}
    except Exception as e:
        return {"valid_format": True, "wellformed": False, "num_asserts": n_asserts, "correct_asserts": 0, "incorrect_asserts": n_asserts, "gold_pass_rate": 0.0, "error": f"{type(e).__name__}: {e}", "results": []}


def compute_stage1_reward(metrics: Dict[str, Any]) -> Tuple[float, str]:
    if not metrics.get("valid_format", False): return INVALID_REWARD, "invalid_format"
    if int(metrics.get("num_asserts", 0)) == 0: return INVALID_REWARD, "zero_asserts"
    if not metrics.get("wellformed", False): return INVALID_REWARD, "not_wellformed"
    if int(metrics.get("num_asserts", 0)) < MIN_ASSERTS_REQUIRED: return SHORT_SUITE_REWARD, "too_few_asserts"
    gold_pass_rate = float(metrics.get("gold_pass_rate", 0.0))
    correct_asserts = int(metrics.get("correct_asserts", 0))
    reward = GOLD_PASS_SCALE * gold_pass_rate + CORRECT_ASSERT_BONUS * min(correct_asserts, CORRECT_ASSERT_BONUS_CAP)
    return float(reward), "gold_pass_reward"


def testgen_gold_reward(prompts: List[str], completions, completions_ids=None, task_id: List[str] = None, question_id: List[str] = None, problem_description: List[str] = None, starter_code: List[str] = None, gold_completion: List[str] = None, entry_point: List[str] = None, reference_test: List[str] = None, prompt_token_len: List[int] = None, trainer_state=None, **kwargs) -> List[float]:
    norm_completions = normalize_completions(completions)
    m = len(prompts) if prompts is not None else 0; n = len(norm_completions)
    if m <= 0: return [0.0] * n
    k = max(1, n // m)
    def prompt_index(i: int) -> int: return min(m - 1, i // k)
    global_step = trainer_state.get("global_step", None) if isinstance(trainer_state, dict) else None
    rewards: List[float] = []; batch_metrics = defaultdict(list); rollout_counter = defaultdict(int)
    tokenizer = kwargs.get("tokenizer", None) or kwargs.get("processing_class", None)
    for i, comp in enumerate(norm_completions):
        j = prompt_index(i)
        ex_task_id = task_id[j] if task_id else ""; ex_question_id = question_id[j] if question_id else ""
        ex_prompt_text = problem_description[j] if problem_description else ""
        ex_gold_code = str(gold_completion[j] if gold_completion else "")
        ex_entry_point = entry_point[j] if entry_point else ""
        key = (ex_task_id, ex_question_id); rollout_idx = rollout_counter[key]; rollout_counter[key] += 1
        attempt_id = f"{ex_task_id}:{ex_question_id}:e{CURRENT_EPOCH}:g{global_step}:r{rollout_idx}"
        gen_only = extract_generated_only_text(tokenizer=tokenizer, prompt_text=prompts[j], completion_text=comp, completion_ids=(completions_ids[i] if completions_ids is not None else None), max_prompt_tokens=MAX_PROMPT_TOKENS) if tokenizer is not None else _strip_think_blocks(_strip_prompt_echo_textwise(comp, prompts[j]))
        extracted = extract_python_block(gen_only)
        salvage_info = salvage_clean_test_code(gen_only, extracted)
        clean_test_code = salvage_info["clean_test_code"]
        if salvage_info.get("parse_error"):
            metrics = {"valid_format": looks_like_check_function(clean_test_code), "wellformed": False, "num_asserts": 0, "correct_asserts": 0, "incorrect_asserts": 0, "gold_pass_rate": 0.0, "error": salvage_info.get("parse_error", ""), "results": []}
        else:
            metrics = evaluate_generated_tests_on_gold(ex_prompt_text, ex_gold_code, clean_test_code, ex_entry_point)
        reward, reward_reason = compute_stage1_reward(metrics)
        rewards.append(reward)
        vals = {"reward": reward, "valid_format": 1.0 if metrics.get("valid_format", False) else 0.0, "wellformed": 1.0 if metrics.get("wellformed", False) else 0.0, "num_asserts": int(metrics.get("num_asserts", 0)), "correct_asserts": int(metrics.get("correct_asserts", 0)), "incorrect_asserts": int(metrics.get("incorrect_asserts", 0)), "gold_pass_rate": float(metrics.get("gold_pass_rate", 0.0)), "short_suite": 1.0 if (int(metrics.get("num_asserts", 0)) > 0 and int(metrics.get("num_asserts", 0)) < MIN_ASSERTS_REQUIRED) else 0.0, "parse_error": 1.0 if salvage_info.get("parse_error") else 0.0}
        for name, value in vals.items(): batch_metrics[name].append(value)
        append_jsonl({"event": "testgen_attempt", "attempt_id": attempt_id, "task_id": ex_task_id, "question_id": ex_question_id, "reward": float(reward), "reward_reason": reward_reason, "valid_format": bool(metrics.get("valid_format", False)), "wellformed": bool(metrics.get("wellformed", False)), "num_asserts": vals["num_asserts"], "correct_asserts": vals["correct_asserts"], "incorrect_asserts": vals["incorrect_asserts"], "gold_pass_rate": vals["gold_pass_rate"], "short_suite": bool(vals["short_suite"]), "min_asserts_required": MIN_ASSERTS_REQUIRED, "epoch": CURRENT_EPOCH, "global_step": global_step, "gen_index": i, "prompt_index": j, "rollout_index": rollout_idx, "num_generations": NUM_GENERATIONS, "prompt_token_len": int(prompt_token_len[j]) if prompt_token_len else None, "salvage_mode": salvage_info.get("salvage_mode", ""), "salvage_parse_error": salvage_info.get("parse_error", ""), "num_salvaged_helper_lines": int(salvage_info.get("num_salvaged_helper_lines", 0)), "num_salvaged_assert_blocks": int(salvage_info.get("num_salvaged_assert_blocks", 0)), "generation_error": metrics.get("error", ""), "prompt_text_full": prompts[j], "generated_text_full": comp, "generated_only_text_full": gen_only, "generated_test_code_raw_extracted": extracted, "generated_test_code_clean": clean_test_code, "gold_code_full": ex_gold_code, "entry_point": ex_entry_point})
    if is_main_process():
        try:
            wandb.log({"reward/mean": float(np.mean(batch_metrics["reward"])) if batch_metrics["reward"] else 0.0, "reward/std": float(np.std(batch_metrics["reward"])) if batch_metrics["reward"] else 0.0, "reward/max": float(np.max(batch_metrics["reward"])) if batch_metrics["reward"] else 0.0, "reward/p95": float(np.percentile(batch_metrics["reward"], 95)) if batch_metrics["reward"] else 0.0, "format/valid_rate": float(np.mean(batch_metrics["valid_format"])) if batch_metrics["valid_format"] else 0.0, "format/wellformed_rate": float(np.mean(batch_metrics["wellformed"])) if batch_metrics["wellformed"] else 0.0, "tests/num_asserts_mean": float(np.mean(batch_metrics["num_asserts"])) if batch_metrics["num_asserts"] else 0.0, "tests/correct_asserts_mean": float(np.mean(batch_metrics["correct_asserts"])) if batch_metrics["correct_asserts"] else 0.0, "tests/incorrect_asserts_mean": float(np.mean(batch_metrics["incorrect_asserts"])) if batch_metrics["incorrect_asserts"] else 0.0, "tests/short_suite_rate": float(np.mean(batch_metrics["short_suite"])) if batch_metrics["short_suite"] else 0.0, "gold/pass_rate_mean": float(np.mean(batch_metrics["gold_pass_rate"])) if batch_metrics["gold_pass_rate"] else 0.0, "format/parse_error_rate": float(np.mean(batch_metrics["parse_error"])) if batch_metrics["parse_error"] else 0.0, "epoch": CURRENT_EPOCH, "trainer/global_step": global_step})
            try: wandb.log({"hist/reward": wandb.Histogram(np.array(batch_metrics["reward"], dtype=np.float32), num_bins=20), "hist/num_asserts": wandb.Histogram(np.array(batch_metrics["num_asserts"], dtype=np.float32), num_bins=20)})
            except Exception: pass
        except Exception as e: print(f"[WARN] wandb logging failed in reward fn: {e}")
        for key in EPOCH_BUF: EPOCH_BUF[key].extend(batch_metrics[key])
    return rewards


@torch.no_grad()
def run_heldout_eval(model, tokenizer, rows: List[Dict[str, Any]], max_items: int = 64) -> Dict[str, float]:
    model_ = model.module if hasattr(model, "module") else model; model_.eval()
    take = rows[:max_items] if max_items and len(rows) > max_items else rows
    if not take: return {"eval/n": 0.0}
    rewards=[]; valid=[]; wellformed=[]; num_asserts_list=[]; correct_list=[]; incorrect_list=[]; pass_rates=[]; short_list=[]
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    for ex in take:
        inputs = tokenizer(ex["prompt"], return_tensors="pt").to(next(model_.parameters()).device)
        gen = model_.generate(**inputs, max_new_tokens=MAX_COMPLETION_TOKENS, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        text = _strip_think_blocks(tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
        extracted = extract_python_block(text); salvage_info = salvage_clean_test_code(text, extracted); clean = salvage_info["clean_test_code"]
        metrics = {"valid_format": looks_like_check_function(clean), "wellformed": False, "num_asserts": 0, "correct_asserts": 0, "incorrect_asserts": 0, "gold_pass_rate": 0.0} if salvage_info.get("parse_error") else evaluate_generated_tests_on_gold(ex.get("problem_description", ""), ex.get("gold_completion", ""), clean, ex.get("entry_point", ""))
        reward, _ = compute_stage1_reward(metrics)
        n_asserts = int(metrics.get("num_asserts", 0))
        rewards.append(reward); valid.append(1.0 if metrics.get("valid_format", False) else 0.0); wellformed.append(1.0 if metrics.get("wellformed", False) else 0.0); num_asserts_list.append(n_asserts); correct_list.append(int(metrics.get("correct_asserts", 0))); incorrect_list.append(int(metrics.get("incorrect_asserts", 0))); pass_rates.append(float(metrics.get("gold_pass_rate", 0.0))); short_list.append(1.0 if (n_asserts > 0 and n_asserts < MIN_ASSERTS_REQUIRED) else 0.0)
    return {"eval/reward_mean": float(np.mean(rewards)), "eval/reward_p95": float(np.percentile(rewards, 95)), "eval/valid_format_rate": float(np.mean(valid)), "eval/wellformed_rate": float(np.mean(wellformed)), "eval/num_asserts_mean": float(np.mean(num_asserts_list)), "eval/correct_asserts_mean": float(np.mean(correct_list)), "eval/incorrect_asserts_mean": float(np.mean(incorrect_list)), "eval/gold_pass_rate_mean": float(np.mean(pass_rates)), "eval/short_suite_rate": float(np.mean(short_list)), "eval/n": float(len(take))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO train Qwen 4B unit-test generator with gold-pass reward and min-10 assert gate.")
    p.add_argument("--model_id", type=str, default=MODEL_ID); p.add_argument("--dataset_name", type=str, default=DATASET_NAME); p.add_argument("--dataset_split", type=str, default=DATASET_SPLIT); p.add_argument("--output_dir", type=str, default=OUTPUT_DIR); p.add_argument("--tester_path", type=str, default=TESTER_PATH); p.add_argument("--exclude_task_ids_json", type=str, default=EXCLUDE_TASK_IDS_JSON); p.add_argument("--max_train_problems", type=int, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS); p.add_argument("--num_generations", type=int, default=NUM_GENERATIONS); p.add_argument("--batch_size", type=int, default=BATCH_SIZE); p.add_argument("--grad_accum_steps", type=int, default=GRAD_ACCUM_STEPS); p.add_argument("--generation_batch_size", type=int, default=GENERATION_BATCH_SIZE)
    p.add_argument("--max_prompt_tokens", type=int, default=MAX_PROMPT_TOKENS); p.add_argument("--max_completion_tokens", type=int, default=MAX_COMPLETION_TOKENS); p.add_argument("--temperature", type=float, default=TEMPERATURE); p.add_argument("--top_p", type=float, default=TOP_P); p.add_argument("--top_k", type=int, default=TOP_K)
    p.add_argument("--lr", type=float, default=LR); p.add_argument("--beta", type=float, default=BETA); p.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO); p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY); p.add_argument("--clip_grad_norm", type=float, default=CLIP_GRAD_NORM)
    p.add_argument("--seed", type=int, default=SEED); p.add_argument("--eval_fraction", type=float, default=EVAL_FRACTION); p.add_argument("--num_fewshot", type=int, default=NUM_FEWSHOT); p.add_argument("--min_asserts_required", type=int, default=MIN_ASSERTS_REQUIRED)
    p.add_argument("--save_steps", type=int, default=SAVE_EVERY_STEPS); p.add_argument("--save_total_limit", type=int, default=SAVE_TOTAL_LIMIT); p.add_argument("--log_every", type=int, default=LOG_EVERY)
    p.add_argument("--wandb_project", type=str, default="TestGen-GRPO"); p.add_argument("--wandb_name", type=str, default="qwen3_4b_gold_pass_min10_grpo"); p.add_argument("--bf16", action="store_true", default=False); p.add_argument("--fp16", action="store_true", default=True)
    return p.parse_args()


def apply_runtime_args(args: argparse.Namespace) -> None:
    global MODEL_ID, DATASET_NAME, DATASET_SPLIT, OUTPUT_DIR, TESTER_PATH, EXCLUDE_TASK_IDS_JSON, CURRICULUM_JSONL, CURRICULUM_STATS, LOG_JSONL, WANDB_RUN_ID_FILE, MAX_TRAIN_PROBLEMS, EPOCHS, NUM_GENERATIONS, BATCH_SIZE, GRAD_ACCUM_STEPS, GENERATION_BATCH_SIZE, MAX_PROMPT_TOKENS, MAX_COMPLETION_TOKENS, TEMPERATURE, TOP_P, TOP_K, LR, BETA, WARMUP_RATIO, WEIGHT_DECAY, CLIP_GRAD_NORM, SEED, EVAL_FRACTION, NUM_FEWSHOT, MIN_ASSERTS_REQUIRED, SAVE_EVERY_STEPS, SAVE_TOTAL_LIMIT, LOG_EVERY
    MODEL_ID=args.model_id; DATASET_NAME=args.dataset_name; DATASET_SPLIT=args.dataset_split; OUTPUT_DIR=args.output_dir; TESTER_PATH=args.tester_path; EXCLUDE_TASK_IDS_JSON=args.exclude_task_ids_json
    CURRICULUM_JSONL=os.path.join(OUTPUT_DIR,"training_testgen_gold_plan.jsonl"); CURRICULUM_STATS=os.path.join(OUTPUT_DIR,"training_testgen_gold_stats.json"); LOG_JSONL=os.path.join(OUTPUT_DIR,"training_testgen_gold_attempt_logs.jsonl"); WANDB_RUN_ID_FILE=os.path.join(OUTPUT_DIR,"wandb_run_id.txt")
    MAX_TRAIN_PROBLEMS=args.max_train_problems; EPOCHS=args.epochs; NUM_GENERATIONS=args.num_generations; BATCH_SIZE=args.batch_size; GRAD_ACCUM_STEPS=args.grad_accum_steps; GENERATION_BATCH_SIZE=args.generation_batch_size; MAX_PROMPT_TOKENS=args.max_prompt_tokens; MAX_COMPLETION_TOKENS=args.max_completion_tokens; TEMPERATURE=args.temperature; TOP_P=args.top_p; TOP_K=args.top_k; LR=args.lr; BETA=args.beta; WARMUP_RATIO=args.warmup_ratio; WEIGHT_DECAY=args.weight_decay; CLIP_GRAD_NORM=args.clip_grad_norm; SEED=args.seed; EVAL_FRACTION=args.eval_fraction; NUM_FEWSHOT=args.num_fewshot; MIN_ASSERTS_REQUIRED=args.min_asserts_required; SAVE_EVERY_STEPS=args.save_steps; SAVE_TOTAL_LIMIT=args.save_total_limit; LOG_EVERY=args.log_every


def main() -> None:
    args_cli = parse_args(); apply_runtime_args(args_cli); setup_logging(OUTPUT_DIR)
    global _tester, resolve_entry_point, eval_all_asserts_with_output, eval_single_assert_with_output, extract_assert_texts, count_asserts
    _tester = import_tester(TESTER_PATH); resolve_entry_point = _tester.resolve_entry_point; eval_all_asserts_with_output = _tester.eval_all_asserts_with_output; eval_single_assert_with_output = _tester.eval_single_assert_with_output; extract_assert_texts = _tester.extract_assert_texts; count_asserts = _tester.count_asserts
    append_jsonl({"event":"script_invocation", "argv":sys.argv, "cwd":os.getcwd(), "model_id":MODEL_ID, "dataset_name":DATASET_NAME, "dataset_split":DATASET_SPLIT, "output_dir":OUTPUT_DIR, "tester_path":TESTER_PATH, "exclude_task_ids_json":EXCLUDE_TASK_IDS_JSON, "min_asserts_required":MIN_ASSERTS_REQUIRED})
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_rows, eval_rows = load_training_rows(tokenizer, DATASET_NAME, DATASET_SPLIT, MAX_TRAIN_PROBLEMS, EXCLUDE_TASK_IDS_JSON)
    if not train_rows: raise RuntimeError("No train rows were built. Check dataset fields, prompt length, and split.")
    train_ds = rows_to_dataset(train_rows)
    global EVAL_ROWS; EVAL_ROWS = eval_rows
    dtype = torch.bfloat16 if args_cli.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, cache_dir=CACHE_DIR, device_map=None)
    if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model, "config"): model.config.use_cache = False
    lora_cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    grpo_args = GRPOConfig(output_dir=OUTPUT_DIR, learning_rate=LR, weight_decay=WEIGHT_DECAY, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=GRAD_ACCUM_STEPS, num_train_epochs=EPOCHS, max_grad_norm=CLIP_GRAD_NORM, warmup_ratio=WARMUP_RATIO, save_safetensors=True, max_prompt_length=MAX_PROMPT_TOKENS, num_generations=NUM_GENERATIONS, generation_batch_size=GENERATION_BATCH_SIZE, max_completion_length=MAX_COMPLETION_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, logging_steps=LOG_EVERY, logging_dir=os.path.join(OUTPUT_DIR,"logs"), log_completions=False, loss_type="grpo", beta=BETA, scale_rewards="batch", fp16=bool(args_cli.fp16 and not args_cli.bf16), bf16=bool(args_cli.bf16), remove_unused_columns=False, save_strategy="steps", save_steps=SAVE_EVERY_STEPS, save_total_limit=SAVE_TOTAL_LIMIT, ddp_find_unused_parameters=False, report_to=["wandb"])
    for name, value in [("top_k", TOP_K), ("min_p", MIN_P), ("presence_penalty", PRESENCE_PENALTY)]:
        try:
            if hasattr(grpo_args, name): setattr(grpo_args, name, value)
        except Exception: pass
    if is_main_process():
        prev_run_id = None
        if os.path.exists(WANDB_RUN_ID_FILE):
            with open(WANDB_RUN_ID_FILE, encoding="utf-8") as f: prev_run_id = f.read().strip() or None
        wandb.init(project=args_cli.wandb_project, name=args_cli.wandb_name, id=prev_run_id, resume="allow", config={"script_path":SCRIPT_PATH, "model":MODEL_ID, "dataset_name":DATASET_NAME, "dataset_split":DATASET_SPLIT, "output_dir":OUTPUT_DIR, "exclude_task_ids_json":EXCLUDE_TASK_IDS_JSON, "epochs":EPOCHS, "num_generations":NUM_GENERATIONS, "batch_size_groups":BATCH_SIZE, "generation_batch_size":GENERATION_BATCH_SIZE, "lr":LR, "beta":BETA, "temperature":TEMPERATURE, "top_p":TOP_P, "top_k":TOP_K, "grad_accum_steps":GRAD_ACCUM_STEPS, "clip_grad_norm":CLIP_GRAD_NORM, "train_rows":len(train_rows), "eval_rows":len(eval_rows), "min_asserts_required":MIN_ASSERTS_REQUIRED, "invalid_reward":INVALID_REWARD, "short_suite_reward":SHORT_SUITE_REWARD, "gold_pass_scale":GOLD_PASS_SCALE, "correct_assert_bonus":CORRECT_ASSERT_BONUS, "correct_assert_bonus_cap":CORRECT_ASSERT_BONUS_CAP, "num_fewshot":NUM_FEWSHOT, "lora_r":LORA_R, "lora_alpha":LORA_ALPHA, "lora_dropout":LORA_DROPOUT, "lora_targets":",".join(LORA_TARGET_MODULES)})
        try:
            if prev_run_id is None and wandb.run and wandb.run.id:
                with open(WANDB_RUN_ID_FILE, "w", encoding="utf-8") as f: f.write(wandb.run.id)
        except Exception as e: logging.warning("Could not persist W&B run id: %s", e)
    def reward_with_tok(*a, **kw):
        kw["tokenizer"] = tokenizer
        return testgen_gold_reward(*a, **kw)
    trainer = GRPOTrainer(model=model, reward_funcs=reward_with_tok, args=grpo_args, train_dataset=train_ds, processing_class=tokenizer, callbacks=[EpochTrackerCallback(), WandbTrainingCallback(), (lambda tok: type("CB", (EpochEndLoggerAndEvalCallback,), {"on_epoch_end": lambda self, *a, **kw: EpochEndLoggerAndEvalCallback.on_epoch_end(self, *a, tokenizer=tok, **kw)}))(tokenizer)()], peft_config=lora_cfg)
    logging.info("Train rows: %d | eval rows: %d | per-device batch=%d | generations=%d", len(train_ds), len(eval_rows), BATCH_SIZE, NUM_GENERATIONS)
    append_jsonl({"event":"training_start", "train_rows":len(train_rows), "eval_rows":len(eval_rows), "epochs":EPOCHS, "num_generations":NUM_GENERATIONS, "batch_size_groups":BATCH_SIZE, "generation_batch_size":GENERATION_BATCH_SIZE, "max_prompt_tokens":MAX_PROMPT_TOKENS, "max_completion_tokens":MAX_COMPLETION_TOKENS, "temperature":TEMPERATURE, "top_p":TOP_P, "top_k":TOP_K, "exclude_task_ids_json":EXCLUDE_TASK_IDS_JSON, "min_asserts_required":MIN_ASSERTS_REQUIRED, "reward_formula":"invalid=-200, zero_asserts=-200, asserts<min=0, else 1000*gold_pass_rate + 20*min(correct_asserts,10)"})
    resume_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if resume_ckpt:
        logging.info("[resume] Resuming from checkpoint: %s", resume_ckpt); append_jsonl({"event":"resume_from_checkpoint", "path":resume_ckpt})
    else:
        logging.info("[resume] No checkpoint found. Starting fresh."); append_jsonl({"event":"resume_from_checkpoint", "path":None})
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.model.save_pretrained(OUTPUT_DIR); tokenizer.save_pretrained(OUTPUT_DIR); trainer.save_model(OUTPUT_DIR)
    if trainer.processing_class and hasattr(trainer.processing_class, "save_pretrained"): trainer.processing_class.save_pretrained(OUTPUT_DIR)
    append_jsonl({"event":"training_done", "output_dir":OUTPUT_DIR})
    logging.info("Saved final checkpoint to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
