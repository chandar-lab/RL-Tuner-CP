#!/bin/bash
#SBATCH --array=1-10
#SBATCH --time=1-00:00:00
#SBATCH --account=rrg-bengioy-ad
#SBATCH --mem=16G
#SBATCH --mail-user=daphne.lafleur@mila.quebec
#SBATCH --mail-type=ALL
module load cuda cudnn java

source ../RL_Tuner/rl_tuner_env/bin/activate

unset JAVA_TOOL_OPTIONS

wandb offline

python script.py $SLURM_ARRAY_TASK_ID