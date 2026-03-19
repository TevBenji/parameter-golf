#!/bin/bash
# SubSixteen — RunPod 8xH100 Setup & Training Script
# Run this on your RunPod instance after SSH'ing in.

set -e

echo "=== Step 1: Clone your fork ==="
cd /root/code 2>/dev/null || mkdir -p /root/code && cd /root/code
git clone https://github.com/TevBenji/parameter-golf.git
cd parameter-golf

echo "=== Step 2: Install dependencies ==="
pip install -r requirements.txt

echo "=== Step 3: Download dataset ==="
python3 data/cached_challenge_fineweb.py --variant sp1024

echo "=== Step 4: Run baseline first (sanity check — expect ~1.2244 BPB) ==="
echo "Uncomment the next block to run baseline, or skip to SubSixteen"
# RUN_ID=baseline_verify \
# DATA_PATH=./data/datasets/fineweb10B_sp1024 \
# TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
# VOCAB_SIZE=1024 \
# MAX_WALLCLOCK_SECONDS=600 \
# TRAIN_LOG_EVERY=50 \
# VAL_LOSS_EVERY=200 \
# torchrun --standalone --nproc_per_node=8 train_gpt.py

echo "=== Step 5: Train SubSixteen ==="
RUN_ID=SubSixteen \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=3 \
MLP_MULT=2 \
NUM_SHARED_BLOCKS=3 \
NUM_LOOPS=10 \
LOOP_SIGNAL_RANK=48 \
PROGRESSIVE_LOSS_LOOPS=4,7 \
PROGRESSIVE_LOSS_WEIGHT=0.3 \
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
SEED=1337 \
AUTHOR=TevBenji \
GITHUB_ID=TevBenji \
SUBMISSION_NAME=SubSixteen \
torchrun --standalone --nproc_per_node=8 train_gpt.py

echo "=== Step 6: Collect artifacts ==="
echo "After training completes, copy these files to your submission folder:"
echo "  - logs/SubSixteen.txt -> train.log"
echo "  - submission.json (auto-generated in working dir)"
echo "  - final_model.ternary.ptz (compressed weights)"
echo ""
echo "Then update submission.json with real val_bpb and bytes_total from the log."
echo "Push to your fork and open PR to openai/parameter-golf."
