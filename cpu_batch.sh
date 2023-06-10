#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=main
#SBATCH --mem=8GB

module load PyTorch
source $HOME/.envs/nlp/bin/activate

python main.py
