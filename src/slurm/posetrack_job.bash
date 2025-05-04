#!/bin/bash

#SBATCH --job-name=CrewExpendable
#SBATCH --account=project_141

#SBATCH --partition=gputest

#SBATCH --time=00:15:00             # hh:mm:ss (optionally append days-)

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # Intel Xeon 2.1 GHz

#SBATCH --mem=32G                   # K,M,G,T

#SBATCH --gres=gpu:v100:1           # NVIDIA Volta V100

module purge
module load pytorch/2.0

source /projappl/project_141/venv_141/bin/activate

srun python3 /scratch/project_141/python/run_posetrack.py


