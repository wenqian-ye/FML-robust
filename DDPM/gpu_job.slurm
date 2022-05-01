#!/bin/bash
#SBATCH --job-name=POSE_CILVR
#SBATCH --open-mode=append
#SBATCH --output=./job_logs/%j_%x.out
#SBATCH --error=./job_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4

singularity exec --nv --overlay $SCRATCH/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH

conda activate hw1_dagger

nvidia-smi

export OMP_NUM_THREADS=1

python train.py
"