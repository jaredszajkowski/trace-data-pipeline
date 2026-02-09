#!/bin/bash
# submit_trace_pipeline.sh

# Load email from user's .env file
source /home/${USER}/trace-data-pipeline/.env

# Submit job with email parameter
sbatch --mail-type=ALL --mail-user=${SLURM_EMAIL} trace-data-pipeline.sbatch