#!/bin/bash

#SBATCH --time=1400
#SBATCH --job-name=sdpa-test-runner
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G 
#SBATCH --partition=p_cpu_all
#SBATCH --account=<YOUR-ACCOUNT-HERE>
#SBATCH --output=/dev/null

# --------------------------------------------------------------------------------------
# modules
# --------------------------------------------------------------------------------------
module load python/3.11.9
VENV="gputest" module load uv
 
# --------------------------------------------------------------------------------------
# run script
# --------------------------------------------------------------------------------------
uv run python -m src.main --config $@ &
wait $!
