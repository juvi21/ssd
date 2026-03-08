# Kimi Linear Handoff

This repo now contains an in-progress Kimi Linear integration. The next agent or engineer should treat this as a handoff for running Kimi on a larger machine and for finishing speculative decoding support.

## Install

Run from the repo root:

```bash
uv sync --extra scripts
```

Required environment variables:

```bash
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
export SSD_CUDA_ARCH=12.0
```

Notes:
- `SSD_HF_CACHE` must point to the Hugging Face `hub/` directory, not its parent.
- On Blackwell GPUs observed in testing here, `SSD_CUDA_ARCH=12.0` was correct.
- Use a real script file when launching multiprocessing tests. Do not use `python - <<'PY' ...` because `spawn` workers fail on `/<stdin>`.

## Model Download

The public checkpoint used in testing:

```text
moonshotai/Kimi-Linear-48B-A3B-Instruct
revision: e1df551a447157d4658b573f9a695d57658590e9
```

Example download:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="moonshotai/Kimi-Linear-48B-A3B-Instruct",
    cache_dir="/path/to/huggingface/hub",
    allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
)
PY
```

Local snapshot path used during validation:

```text
/workspace/.cache/huggingface/hub/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/snapshots/e1df551a447157d4658b573f9a695d57658590e9
```

## What Was Verified

- Kimi AR now works end to end on 4 GPUs after two loader/TP fixes described below.
- A short AR sanity check generated coherent English output.
- A larger AR benchmark also completed successfully.

AR benchmark used:

```bash
uv run python scripts/bench_kimi.py \
  --model /path/to/kimi/snapshot \
  --mode ar \
  --gpus 4 \
  --num-prompts 8 \
  --input-len 128 \
  --output-len 128 \
  --batch-size 2 \
  --max-model-len 2048 \
  --block-size 128 \
  --gpu-mem-util 0.2
```

Observed result on the 4x RTX PRO 6000 Blackwell box:

```text
generated_tokens: 1024
elapsed_s: 43.888
throughput_tok_s: 23.33
prefill throughput: 121 tok/s
decode throughput: 28 tok/s
```

## Fixes Applied In This Worktree

1. KDA short-conv TP loading
- Kimi KDA `ShortConvolution` weights were not sharded on TP load.
- This caused AR model load to fail with shape mismatch on 4 GPUs.
- Fixed by attaching a TP-aware loader in `ssd/models/kimi_linear.py`.

2. Uneven vocab sharding
- Kimi vocab size is not guaranteed to divide every TP size used by async SSD target splits.
- Fixed `ssd/layers/embed_head.py` to support uneven embedding / LM-head shards with padded gather.

These fixes are necessary for AR and for some future speculative work, but they are not sufficient to make SSD work on 4 GPUs.

## Main Findings About Why Kimi SSD Does Not Work On 4 GPUs Yet

### 1. Async SSD topology becomes 3 target GPUs + 1 draft GPU

In this repo, async SSD reserves one GPU for the draft and uses `num_gpus - 1` for the target.

With `--gpus 4`, Kimi async becomes:
- target TP size = 3
- draft TP size = 1

This is different from the README examples for large Llama/Qwen async SSD, which use 5 GPUs total.

### 2. Kimi dimensions do not fit a 3-way TP split under current assumptions

Current linear / TP code assumes exact divisibility for tensor-parallel splits.

Kimi uses dimensions like:
- vocab size `163840`
- attention heads `32`
- KDA heads `32`
- KDA projection size `32 * 128 = 4096`
- MoE intermediate size `1024`

Several of these do not divide cleanly by `3`, so 4-total-GPU async SSD fails during target init on the 3-way split.

### 3. Draft model falls back to the full target model

If no separate Kimi-family draft is configured, the code falls back to:

```text
draft = target
```

That means the draft is the full `Kimi-Linear-48B-A3B-Instruct` checkpoint.

### 4. Draft is still single-GPU only

`DraftRunner` is currently constructed with `num_tp_gpus=1`.

So even on a larger machine, async SSD still needs the draft checkpoint to fit on one GPU unless draft TP support is implemented.

### 5. On 3 total GPUs, the draft OOMs

Using 2 target GPUs + 1 draft GPU avoids the 3-way target split issue, but the single-GPU draft OOMs with the full Kimi 48B checkpoint on a 96 GB card.

## What Needs To Be Implemented For Kimi SSD

Minimum path:

1. Provide a real smaller Kimi-family draft
- Distilled Kimi draft
- Quantized Kimi draft
- Or implement draft TP so the draft can span multiple GPUs

2. Add uneven TP support beyond the vocab layer
- `ColumnParallelLinear`
- `RowParallelLinear`
- Kimi head partitioning
- KDA projections and loaders
- Any assumptions in KV/cache layout and logits gather

3. Re-run exactness checks at `temperature=0`
- AR vs sync speculative
- AR vs async SSD
- Match token IDs exactly before benchmarking

## Recommended Test Plan On A More Powerful Machine

### Option A: Fastest path to meaningful SSD

Use:
- 4 GPUs for target
- 1 separate GPU for a smaller Kimi draft

Then run:

```bash
uv run python scripts/bench_kimi.py \
  --model /path/to/kimi/target \
  --draft /path/to/kimi/draft \
  --mode async \
  --gpus 5 \
  --k 4 \
  --f 3 \
  --batch-size 1 \
  --num-prompts 8 \
  --input-len 128 \
  --output-len 128 \
  --max-model-len 2048 \
  --block-size 128 \
  --gpu-mem-util 0.2
```

### Option B: If only the 48B checkpoint is available

Implement draft TP first, or async SSD will still be blocked by the full-size single-GPU draft.

### Option C: Exactness harness

Run short deterministic prompts with:
- `--mode ar`
- `--mode sync`
- `--mode async`

Compare `token_ids` exactly at `temperature=0.0`.

## Practical Notes

- Kimi forces eager mode and disables prefix cache in current config handling.
- Expect long model load / warmup times even for short prompts.
- The benchmark script `scripts/bench_kimi.py` is available in this worktree and is the easiest entrypoint for repeated Kimi runs.

## Bottom Line

- AR is working.
- 4-total-GPU async SSD is not currently a valid topology for Kimi in this implementation.
- More GPUs help only if:
  - the target can use a divisible TP size, and
  - the draft is smaller or multi-GPU.

## Extra Observation

On a 5-GPU machine, the first meaningful Kimi SSD comparison is:

- SSD on 5 total GPUs = 4 target GPUs + 1 draft GPU
- versus AR on 4 target GPUs

Do not treat "AR on 5 GPUs" as the main baseline for Kimi in the current code. A 5-way target TP split is not a valid Kimi topology under the repo's current equal-shard assumptions, while async SSD on 5 total GPUs gives the target a 4-way TP split, which is valid for Kimi's current dimensions.

Whether SSD is actually faster than AR on that machine is still empirical:

- likely yes only if the Kimi draft is much cheaper than the 48B target and acceptance/cache-hit rates are good
- not guaranteed if the draft is too heavy or the acceptance rate is poor
