#!/bin/bash

################
# Slurm settings
################
#SBATCH --job-name=synthwave_preprocessing            # Job name
#SBATCH --mail-type=FAIL                              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-type=END
#SBATCH --mail-user=h.p.rice@leeds.ac.uk              # Where to send mail
#SBATCH --array=1                                     # Number of runs, --array=1-X will run X jobs (X >= 1)
#SBATCH --ntasks=1                                    # Number of tasks to run, change as desired
#SBATCH --cpus-per-task=2                             # Number of CPU cores per task
#SBATCH --mem=64gb                                   # Job memory request
#SBATCH --time=01:00:00                               # Time limit hrs:min:sec
#SBATCH --output=logs/logs/batch-%A-%a.out
#SBATCH --error=logs/errors/batch-%A-%a.err


echo -e "\nRunning Synthwave pre-processing steps... \n  Source data path: $2\n  Subset fraction: $4\n  Emails to: $6\n  Time: $8"
echo -e "Task $SLURM_JOB_ID"
echo -e "Running with $SLURM_CPUS_PER_TASK CPU cores, $SLURM_CPUS_ON_NODE CPU cores per node"
echo -e "Running task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX\n"

export MAXIT=1  # Testing
export NCORES=2  # Testing

#python src/synthwave/utils/uk/pre_process.py "$2"
#Rscript src/synthwave/synthesizer/imputation/adults_imputation.R "$2" -f "$4" -n $NCORES -m $MAXIT  # Testing
Rscript src/synthwave/synthesizer/imputation/adults_imputation.R "$2" -f "$4" -n $SLURM_CPUS_PER_TASK -m $MAXIT
#Rscript src/synthwave/synthesizer/imputation/adults_imputation.R "$2" -f "$4" -n $SLURM_CPUS_ON_NODE -m $MAXIT
# python src/synthwave/synthesizer/correct_and_train.py "$2"

# If no errors...
exit 0
