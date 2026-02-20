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
sbatch --mail-type=ALL --mail-user=${SLURM_EMAIL} trace-data-pipeline.sbatch

# STAGE 0 Part A: TRACE Enhanced, Standard, 144A Data Extraction
# JOB1=$(sbatch --parsable --mail-type=ALL --mail-user=${SLURM_EMAIL} trace-data-pipeline_stage0_partA.sbatch)
# echo "STAGE 0 Part A: TRACE Enhanced, Standard, 144A Data Extraction: $JOB1"

# STAGE 0 Part B: TRACE Enhanced Process, Build Data Reports
# STAGE 1: Daily Aggregation & Analytics
# JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 --mail-type=ALL --mail-user=${SLURM_EMAIL} trace-data-pipeline_stage0_partB_stage1.sbatch)
# echo "STAGE 0 Part B: TRACE Enhanced Process, Build Data Reports and STAGE 1: Daily Aggregation & Analytics: $JOB2 (depends on $JOB1)"
