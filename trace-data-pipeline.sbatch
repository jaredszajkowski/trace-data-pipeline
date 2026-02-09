#!/bin/bash
#SBATCH --job-name=trace-data-pipeline
#SBATCH --output=%j_trace-data-pipeline.out
#SBATCH --error=%j_trace-data-pipeline.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=build
#SBATCH --time=6:00:00
#SBATCH --account=finm32900

# Change to the directory for trace-data-pipeline
cd /home/jszajkowski/trace-data-pipeline
echo "Changed to trace-data-pipeline directory"

# Activate Python virtual environment
source /project/finm32900/tdp-venv-p3119/bin/activate
echo "Activated virtual environment"

# Run the pipeline
echo "Executing pipeline"
./run_pipeline.sh
