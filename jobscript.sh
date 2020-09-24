#!/usr/local_rwth/bin/zsh


# Job configuration ---

#SBATCH --job-name=fcn_configxxxx
#SBATCH --output=results/job.%j.log

## OpenMP settings
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

## Request for a node with 2 Tesla P100 GPUs
#SBATCH --gres=gpu:pascal:2

#SBATCH --time=01:00:00


# Load CUDA 
module load cuda

# Debug info
echo; echo
nvidia-smi
echo

# Execute script
python_interpreter="../maastro_env/bin/python3"

$python_interpreter training_script.py




#------------------------
# Note: All relative paths are relative to the directory containing the job script