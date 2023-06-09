#!/bin/bash
#SBATCH --time=00:04:00
#SBATCH --ntasks=1
#SBATCH --job-name=main
#SBATCH --mem=8000

module load PyTorch
source $HOME/.envs/nlp/bin/activate

python main.py
