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

cd /fs/ess/PAS2136/fangxun/FloraPalooza/scripts

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
    --epochs 20 \
    --batch_size 4096 \
    --lr 1e-3 \
    --hidden 512 \
    --out_dir output/MLP_512

echo "=== Job finished: $(date) ==="
