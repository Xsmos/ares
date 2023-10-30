#!/bin/bash
#SBATCH -JFIT_PARAM 
#SBATCH --account=gts-jw254-coda20
#SBATCH --mem-per-cpu=12G
#SBATCH -t24:00:00
#SBATCH --ntasks-per-node=1 --cpus-per-task=24
#SBATCH -qinferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bxia34@gatech.edu
#SBATCH --array=0-116

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID."
echo "EXECUTING on the machine:" $(hostname)

cd ares/
srun python fit_param.py
