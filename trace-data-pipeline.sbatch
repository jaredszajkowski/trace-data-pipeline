#!/bin/bash
#SBATCH --job-name=trace-data-pipeline
#SBATCH --output=%j_trace-data-pipeline.out
#SBATCH --error=%j_trace-data-pipeline.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000
#SBATCH --partition=build
#SBATCH --time=6:00:00
#SBATCH --account=finm32900
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jszajkowski@rcc.uchicago.edu

# Change to the directory for trace-data-pipeline
cd /home/jszajkowski/trace-data-pipeline

# Point to .pgpass file in home directory
#export PGPASSFILE=/home/jszajkowski/.pgpass

# Activate Python virtual environment
source /project/finm32900/tdp-venv-p3119/bin/activate

# Test WRDS connectivity
# echo "Testing WRDS connection..."
# echo "PGPASSFILE location: ${PGPASSFILE:-$HOME/.pgpass}"
# ls -la ~/.pgpass
# cat ~/.pgpass  # Verify it's readable (will show in logs)

# Test if compute node can reach WRDS
# ping -c 3 wrds-pgdata.wharton.upenn.edu || echo "Cannot ping WRDS"
# nc -zv wrds-pgdata.wharton.upenn.edu 9737 || echo "Cannot connect to WRDS port 9737"

# Test to see if compute node can see the internet
# ping -c 3 www.google.com || echo "Cannot ping google"

# Run the pipeline
./run_pipeline.sh
