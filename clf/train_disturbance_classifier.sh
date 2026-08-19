#!/bin/bash
#SBATCH --account=pas2136
#SBATCH --job-name=train_disturbance_classifier
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liu.12122@osu.edu
#SBATCH --output=slurm_output/train_disturbance_classifier.out.%j

set -x

cd /fs/ess/PAS2136/fangxun/FloraPalooza/SAtrEes/clf

echo "=== Job started: $(date) ==="

# Step 1: prepare labels
if [ ! -f output/disturbance_labels.npz ]; then
    echo "--- Preparing labels ---"
    python prepare_labels.py --out_dir output
fi

# Step 2: train classifier (default: disturb_recent_class, 10 classes)
echo "--- Training classifier ---"
python train_disturbance_classifier.py \
    --target disturb_recent_class \
    --balanced \
    --exclude_not_mapped \
    --max_years_since 20 \
    --epochs 30 \
    --batch_size 4096 \
    --lr 1e-3 \
    --hidden 1024 \
    --out_dir output/MLP_1024_balanced_no-not-mapped/recent-20-years

echo "=== Job finished: $(date) ==="
