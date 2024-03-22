#!/bin/bash
#SBATCH --job-name=NoticIA_Zero
#SBATCH --cpus-per-task=22
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --output=.slurm/NoticIA_Zero.out.txt
#SBATCH --error=.slurm/NoticIA_Zero.err.txt


source /ikerlariak/igarcia945/envs/pytorch2/bin/activate


export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_ENTITY=igarciaf
export WANDB_PROJECT=No-ClickBait-Project_Models_zero_SEPLN_3
export OMP_NUM_THREADS=16
export WANDB__SERVICE_WAIT=300

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

for config in configs/configs_zero-shot/*; do
    torchrun --standalone --master_port 37229 --nproc_per_node=4 run.py "$config"
done
