#!/bin/bash

#SBATCH --job-name=VAECC
#SBATCH --time=0-0:35:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=1
# output files
#SBATCH -o /data/compoundx/anand/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/anand/job_log/%x-%u-%j.err


module load Anaconda3/2023.03

echo $EBROOTANACONDA3
echo $EBROOTANACONDA3/etc/profile.d/conda.sh

source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate ~/.conda/envs/TORCH21

python s_train_vae.py -a cuda

