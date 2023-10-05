#!/bin/bash

#SBATCH --job-name=VAEM
#SBATCH --time=0-1:35:00
#SBATCH -G nvidia-a100:1
#SBATCH --mem-per-cpu=64G
#SBATCH --cpus-per-task=1
# output files
#SBATCH -o /data/compoundx/anand/job_log/%x-%u-%j.out
#SBATCH -e /data/compoundx/anand/job_log/%x-%u-%j.err

module load Anaconda3/2020.07
source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate TORCH311

# python s1_train_ModalVAE.py -m grouped_weather -b 0.0001 


python s1_train_ModalVAE.py -m grouped_states -b 0.0001