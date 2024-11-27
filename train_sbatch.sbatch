#!/bin/bash
#SBATCH --job-name="4d_zombie"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:36G
#SBATCH --mem=36G
#SBATCH --time=150:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/user/sang/slurm/4d_zombie%j.out
#SBATCH --error=/storage/user/sang/slurm/4d_zombie%j.out

#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="Neurips 07.May"

codepath="/home/wiss/sang/git/Implicit-surf-Deformation/"


python "${codepath}/train.py" --conf "${codepath}conf/faust.conf" --expname faust --subindex 11 --reset