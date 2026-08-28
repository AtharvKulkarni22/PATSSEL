# PATSSEL

Code for **Debug or Regenerate? Test-Guided Selection for LLM Code Repair**.

In this paper we introduce PATSSEL. PATSSEL generates a regenerated candidate and a debugged candidate, evaluates them with verifier tests, and selects the candidate with the higher verifier-test pass rate. This repository contains the training and inference code used for the results in the main table.

## Setup

Python 3.10 was used for the experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Unless overridden with `--seed`, the scripts use the paper seed `1012`.

## Models

The experiments use:

- regeneration: `Qwen/Qwen3-8B`
- debugging: `Qwen/Qwen3-8B` with the trained debugger LoRA
- Base Test Gen: `Qwen/Qwen3-4B-Instruct-2507`
- RL Test Gen: `Qwen/Qwen3-4B-Instruct-2507` with the trained test-generator LoRA

By default, trained adapters are expected at:

```text
checkpoints/debugger/
checkpoints/test_generator/
```

Both paths can be overridden from the command line.

## Data

### LeetCode

`data/leetcode/train/debugger_train.tsv` is the debugger-training source used by the training script. It contains 1,300 source rows. The script deterministically expands these rows into the 3,000 targeted failing-test training instances used for GRPO.

The dev and test files for LeetCode are present here:

```text
data/leetcode/dev/q1_0_25.tsv
data/leetcode/dev/q2_25_50.tsv
data/leetcode/dev/q3_50_75.tsv
data/leetcode/dev/q4_75_99.tsv

data/leetcode/test/q1_0_25.tsv
data/leetcode/test/q2_25_50.tsv
data/leetcode/test/q3_50_75.tsv
data/leetcode/test/q4_75_99.tsv
```

### HE+Fix

HE+Fix is loaded from Hugging Face: `archiki/UTGenDebug`, split `he_plus_fix`. The evaluation contains 158 examples.

### LiveCodeBench

`data/livecodebench/test.jsonl` contains the 813 failing Qwen-generated programs used as repair inputs:

- Qwen3-1.7B: 258
- Qwen3-4B: 196
- Qwen3-8B: 192
- Qwen3-14B: 167

Official evaluation tests are loaded from LiveCodeBench `release_v1` at runtime.

## Train the debugger

```bash
python scripts/train/train_debugger.py \
  --train_file data/leetcode/train/debugger_train.tsv \
  --model_id Qwen/Qwen3-8B \
  --output_dir checkpoints/debugger
```

The training objective is the pass-rate improvement reward from the paper:

```text
1000 * (pass_rate(repair) - pass_rate(buggy_code))
```

## Train the unit-test generator

```bash
python scripts/train/train_test_generator.py \
  --model_id Qwen/Qwen3-4B-Instruct-2507 \
  --dataset_name newfacade/LeetCodeDataset \
  --dataset_split train \
  --exclude_task_ids_json data/leetcode/eval_task_ids.json \
  --output_dir checkpoints/test_generator
```

The reward for the unit-test generator is:

```text
invalid / no assertions   -> -200
fewer than 10 assertions  -> 0
otherwise                 -> 1000 * gold_pass_rate
                             + 20 * min(correct_asserts, 10)
```

## Generate verifier tests

### LeetCode

Base Test Gen:

```bash
python scripts/generate_verifier_tests_leetcode.py \
  --output_dir outputs/verifier_tests/leetcode/base
```

RL Test Gen:

```bash
python scripts/generate_verifier_tests_leetcode.py \
  --peft_model_path checkpoints/test_generator \
  --output_dir outputs/verifier_tests/leetcode/rl
```

The default input is the 400-example test split. Use `--buggy_tsvs` to point to the four dev files instead.

### HE+Fix

Base Test Gen:

```bash
python scripts/generate_verifier_tests_hefix.py \
  --output_dir outputs/verifier_tests/hefix/base
```

RL Test Gen:

```bash
python scripts/generate_verifier_tests_hefix.py \
  --peft_model_path checkpoints/test_generator \
  --output_dir outputs/verifier_tests/hefix/rl
```

### LiveCodeBench

Pass a directory containing the evaluated files named `*_codegeneration_output_eval_all.json` from the Qwen3-1.7B, 4B, 8B, and 14B source runs.

Base Test Gen:

```bash
python scripts/generate_verifier_tests_livecodebench.py \
  --codegen_root data/livecodebench/test.jsonl \
  --reference_root /path/to/evaluated_lcb_qwen_source_runs \
  --output_dir outputs/verifier_tests/livecodebench/base
```

RL Test Gen:

```bash
python scripts/generate_verifier_tests_livecodebench.py \
  --codegen_root data/livecodebench/test.jsonl \
  --reference_root /path/to/evaluated_lcb_qwen_source_runs \
  --peft_model_path checkpoints/test_generator \
  --output_dir outputs/verifier_tests/livecodebench/rl
```

The `reference_root` is an intermediate input to verifier-test generation, not a reported repair output.

## Run the main-table experiments

All three runners use the same nine experiment tags:

| Tag | Verifier | Candidates |
|---|---|---|
| `base_regen` | Base Test Gen | 2 x ReGen |
| `base_debug` | Base Test Gen | 2 x RL Debug |
| `base_patssel` | Base Test Gen | 1 x ReGen + 1 x RL Debug |
| `rl_regen` | RL Test Gen | 2 x ReGen |
| `rl_debug` | RL Test Gen | 2 x RL Debug |
| `rl_patssel` | RL Test Gen | 1 x ReGen + 1 x RL Debug |
| `gold_regen` | Gold tests | 2 x ReGen |
| `gold_debug` | Gold tests | 2 x RL Debug |
| `gold_patssel` | Gold tests | 1 x ReGen + 1 x RL Debug |

**Base / RL / Gold refers to the verifier used for selection, not the repair model.** For example, `base_debug` generates two candidates with the RL-trained debugger and selects between them using Base Test Gen verifier tests.

LeetCode:

```bash
python scripts/run_leetcode.py \
  --all \
  --log_dir outputs/results/leetcode \
  --rl_peft_checkpoint checkpoints/debugger
```

HE+Fix:

```bash
python scripts/run_hefix.py \
  --all \
  --log_dir outputs/results/hefix \
  --rl_peft_checkpoint checkpoints/debugger
```

LiveCodeBench:

```bash
python scripts/run_livecodebench.py \
  --all \
  --log_dir outputs/results/livecodebench \
  --rl_peft_checkpoint checkpoints/debugger
```

Use `--experiments <tag> ...` instead of `--all` to run selected rows.

## Evaluation utilities

Benchmark-specific execution and scoring code is kept under `patssel/evaluation/`.

- `leetcode.py` executes callable LeetCode solutions against assertion-based tests. It is used by LeetCode evaluation and by the training rewards.
- `hefix.py` executes HE+Fix functions against the benchmark input/output tests with timeout and memory limits.
- `livecodebench.py` loads LiveCodeBench `release_v1` tests and evaluates stdin/stdout programs with per-test timeouts and output normalization.

Training scripts are kept under `scripts/train/`. The remaining scripts under `scripts/` handle verifier generation, inference, experiment selection, and table construction. The files under `patssel/evaluation/` contain the benchmark-specific execution logic.

## Rebuild the Qwen main-table rows

After all nine experiments have completed on all three benchmarks:

```bash
python scripts/make_main_table.py
```

This writes:

```text
outputs/main_results_qwen.tsv
outputs/main_results_qwen.tex
```

The table builder reads the non-backtracking results used in the paper. For LeetCode it reports Q1--Q4 and the micro-average from the four released buckets; for LiveCodeBench it reports the four original buggy-pass-rate buckets and the micro-average; HE+Fix reports the overall reference-test accuracy.

## Repository layout

```text
PATSSEL/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── leetcode/
│   │   ├── train/debugger_train.tsv
│   │   ├── dev/{q1_0_25,q2_25_50,q3_50_75,q4_75_99}.tsv
│   │   ├── test/{q1_0_25,q2_25_50,q3_50_75,q4_75_99}.tsv
│   │   ├── eval_task_ids.json
│   │   └── split_manifest.json
│   ├── hefix/README.md
│   └── livecodebench/test.jsonl
├── patssel/
│   └── evaluation/
│       ├── leetcode.py
│       ├── hefix.py
│       └── livecodebench.py
└── scripts/
    ├── train/
    │   ├── train_debugger.py
    │   └── train_test_generator.py
    ├── generate_verifier_tests_leetcode.py
    ├── generate_verifier_tests_hefix.py
    ├── generate_verifier_tests_livecodebench.py
    ├── run_leetcode.py
    ├── run_hefix.py
    ├── run_livecodebench.py
    └── make_main_table.py
```
