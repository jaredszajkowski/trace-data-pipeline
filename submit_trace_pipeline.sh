#!/bin/bash
# submit_trace_pipeline.sh

ENV_FILE="/home/${USER}/trace-data-pipeline/.env"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please create this file with your email address:"
    echo "SLURM_EMAIL=your_email@rcc.uchicago.edu"
    exit 1
fi

# Load email from user's .env file
source "$ENV_FILE"

# Verify email was loaded
if [ -z "$SLURM_EMAIL" ]; then
    echo "Error: SLURM_EMAIL not set in $ENV_FILE"
    exit 1
fi

# Submit job with email parameter
# sbatch --mail-type=ALL --mail-user=${SLURM_EMAIL} trace-data-pipeline.sbatch

# Stage 0: Submit Enhanced, Standard, and 144A TRACE data extraction jobs
# Stage 0: Build data reports after all extraction jobs complete
JOB1=$(sbatch --parsable --mail-type=ALL --mail-user=${SLURM_EMAIL} trace-data-pipeline_stage0.sbatch)
echo "Stage 0: $JOB1"

# Stage 1: Process daily aggregation after stage0 reports are ready
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 --mail-type=ALL --mail-user=${SLURM_EMAIL} trace-data-pipeline_stage1.sbatch)
echo "Stage 1: $JOB2 (depends on $JOB1)"
