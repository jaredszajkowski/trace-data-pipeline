#!/bin/bash
#SBATCH --job-name=trace-data-pipeline-stage1
#SBATCH --output=%j_trace-data-pipeline-stage1.out
#SBATCH --error=%j_trace-data-pipeline-stage1.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=16000
#SBATCH --partition=build
#SBATCH --time=6:00:00
#SBATCH --account=finm32900

# Change to the directory for trace-data-pipeline
cd /home/${USER}/trace-data-pipeline
echo "Changed to trace-data-pipeline directory"

# Activate Python virtual environment
source /project/finm32900/tdp-venv-p3119/bin/activate
echo "Activated virtual environment"

# Run the pipeline
echo "Executing stage 1 pipeline"
./run_pipeline_stage1.sh
