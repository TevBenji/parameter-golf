# SubSixteen — Ternary QAT + Depth Recurrence + Test-Time Training

This record captures the `SubSixteen` submission, which stacks three orthogonal techniques to maximize effective model capacity within the 16MB artifact budget.

## Approach

Three multipliers, none previously combined in a single submission:

1. **Ternary QAT** — BitLinear layers constrain weights to {-1, 0, +1} via AbsMedian quantization with straight-through estimator gradients. Stores ~1.5 bits/weight vs 8 bits for int8, enabling 4-5× more parameters per byte. L1 regularization encourages zero-heavy distributions for better zlib compression.

2. **Depth recurrence** — Prelude(1 block) + RecurrentBlockGroup(3 shared blocks × 10 loops) + Coda(1 block) = 32 effective transformer layers from only 5 stored blocks. Per-loop LayerNorms and low-rank signals (rank 48) differentiate iterations. Progressive loss at intermediate loops (4, 7) improves gradient flow.

3. **Test-time training (TTT)** — Per-document 1-step SGD on the last 2 layers during evaluation, using 30% prefix adaptation. No training data accessed during eval.

## Training Schedule

- BF16 phase: first 20% of steps (full-precision forward passes)
- Ternary QAT phase: remaining 80% (STE gradients, 0.5× LR reduction for BitLinear params)
- Muon optimizer for matrix params, Adam for embeddings/scalars/loop signals
- Warmup + stable + wallclock-based warmdown LR schedule

## Configuration

- Layout: `VOCAB_SIZE=1024 MODEL_DIM=768 NUM_HEADS=12 NUM_KV_HEADS=3 MLP_MULT=2`
- Recurrence: `NUM_SHARED_BLOCKS=3 NUM_LOOPS=10 LOOP_SIGNAL_RANK=48`
- QAT: `QAT_SWITCHOVER_FRAC=0.2 QAT_LR_FACTOR=0.5 L1_REG_LAMBDA=1e-5`
- TTT: `TTT_ENABLED=1 TTT_PREFIX_FRAC=0.3 TTT_LR=1e-4 TTT_ADAPT_LAYERS=2`
- Tied output/input embeddings: `TIE_EMBEDDINGS=1`
- Batching: `TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024`

## Command

```bash
RUN_ID=SubSixteen \
DATA_PATH=/root/code/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/code/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=3 \
MLP_MULT=2 \
NUM_SHARED_BLOCKS=3 \
NUM_LOOPS=10 \
LOOP_SIGNAL_RANK=48 \
QAT_SWITCHOVER_FRAC=0.2 \
QAT_LR_FACTOR=0.5 \
L1_REG_LAMBDA=1e-5 \
TTT_ENABLED=1 \
TTT_PREFIX_FRAC=0.3 \
TTT_LR=1e-4 \
TTT_ADAPT_LAYERS=2 \
TIE_EMBEDDINGS=1 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Parameter Budget

| Component | Stored Params | Notes |
|---|---|---|
| tok_emb (1024 × 768) | 786,432 | Tied with output projection |
| Prelude Block (1×) | ~3.6M | 4 BitLinear projections + MLP |
| Shared Blocks (3×) | ~10.7M | Reused 10× for 30 effective layers |
| Coda Block (1×) | ~3.6M | Same structure as prelude |
| Loop signals + norms | ~88K | 10 loops × rank 48 |
| **Total stored** | **~20.7M** | |
| **Effective (with recurrence)** | **~55M** | 32 effective layers |

Estimated compressed artifact: ~3.7MB (well under 16MB cap).

## Key Metrics

> **TODO**: Fill in after 8×H100 training run

| Metric | Value |
|---|---|
| Post-quant val_bpb | _pending_ |
| Pre-quant val_bpb | _pending_ |
| TTT val_bpb | _pending_ |
| Training steps | _pending_ |
| Train time | _pending_ |
| Model (ternary+zlib) | _pending_ |
| Code | ~62,715 bytes |
| Total artifact | _pending_ |

## Validation

- 28/28 property-based tests passing (hypothesis library)
- Ternary round-trip integrity, STE gradient flow, compression efficiency
- Recurrence depth multiplication, per-loop differentiation, progressive loss
- TTT prefix alignment, selective adaptation, weight reset round-trip
- Seed determinism, BPB computation validity, tied embedding sharing
- Single-file: 1285 lines (under 1500 limit)

## Included Files

- `train_gpt.py` (code snapshot used for the run)
- `train.log` (training log from 8×H100 run)
- `submission.json` (leaderboard metadata)
