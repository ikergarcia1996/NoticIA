#!/bin/bash
#SBATCH --job-name=NoticIA_Train
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --output=.slurm/NoticIA_Train.out.txt
#SBATCH --error=.slurm/NoticIA_Train.err.txt


source /ikerlariak/igarcia945/envs/pytorch2/bin/activate


export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_ENTITY=igarciaf
export WANDB_PROJECT=No-ClickBait-Project_SEPLN_2_train
export OMP_NUM_THREADS=16
export WANDB__SERVICE_WAIT=300

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"


torchrun --standalone --master_port 37231 --nproc_per_node=4 run.py configs/configs_finetune/gemma-2b-it.yaml
torchrun --standalone --master_port 37231 --nproc_per_node=4 run.py configs/configs_finetune/gemma-2b-it_Test.yaml
torchrun --standalone --master_port 37231 --nproc_per_node=4 run.py configs/configs_finetune/Nous-Hermes-2-SOLAR-10.7B.yaml
torchrun --standalone --master_port 37231 --nproc_per_node=4 run.py configs/configs_finetune/Nous-Hermes-2-SOLAR-10.7B_Test.yaml
torchrun --standalone --master_port 37231 --nproc_per_node=4 run.py configs/configs_finetune/openchat-3.5-0106_Test.yaml
torchrun --standalone --master_port 37231 --nproc_per_node=4 run.py configs/configs_finetune/openchat-3.5-0106_Test.yaml


