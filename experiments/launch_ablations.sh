#!/bin/bash
# Launch all OC-CCL ablations across 8 GPUs
# ~8GB per run, 98GB per GPU → pack ~10 per GPU
# Total: 16 ablations across 8 GPUs (2 per GPU)

cd /srv/disk00/dfeng8/work/ijcv/SST
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate sm2

SCRIPT="experiments/oc_ccl_ablation.py"
CKPT="checkpoints/sam2_hiera_large.pt"
EP=10
BASE="outputs/ablation"

run() {
    local name=$1 gpu=$2; shift 2
    echo "Launching $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup python -u $SCRIPT \
        --name "$name" --gpu 0 --checkpoint $CKPT --epochs $EP \
        --output_dir "$BASE/$name" "$@" \
        > "$BASE/$name.log" 2>&1 &
}

mkdir -p $BASE

# ── GPU 0: Learning rate ──
run "lr_1e-4"       0  --lr 1e-4
run "lr_1e-6"       0  --lr 1e-6

# ── GPU 1: Learning rate (cont) + baseline ──
run "lr_5e-5"       1  --lr 5e-5
run "lr_1e-5"       1  --lr 1e-5   # baseline

# ── GPU 2: BCE/Dice weights ──
run "bce_only"      2  --bce_w 1.0 --dice_w 0.0
run "dice_only"     2  --bce_w 0.0 --dice_w 1.0

# ── GPU 3: BCE/Dice weights (cont) ──
run "bce2_dice1"    3  --bce_w 2.0 --dice_w 1.0
run "bce1_dice2"    3  --bce_w 1.0 --dice_w 2.0

# ── GPU 4: LoRA rank ──
run "lora_r4"       4  --lora_rank 4
run "lora_r16"      4  --lora_rank 16

# ── GPU 5: LoRA rank (cont) ──
run "lora_r8"       5  --lora_rank 8
run "lora_r4_lr1e4" 5  --lora_rank 4 --lr 1e-4

# ── GPU 6: Memory reset ──
run "mem_reset"     6  --reset_memory
run "mem_reset_lr1e4" 6  --reset_memory --lr 1e-4

# ── GPU 7: Combos ──
run "lora_r4_dice"  7  --lora_rank 4 --bce_w 0.0 --dice_w 1.0
run "lora_r4_reset" 7  --lora_rank 4 --reset_memory

echo ""
echo "Launched 16 ablations. Monitor with:"
echo "  tail -f $BASE/*.log"
echo "  nvidia-smi"
echo ""
echo "Collect results with:"
echo "  cat $BASE/*/results.json"
