#!/bin/bash

# Initialize modules.
source /etc/profile

# Load Anaconda module.
module load anaconda/2021a

# Setup virtual environment.
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies.
pip install -r ../requirements.txt

# Log task id.
echo "Sweep task ID: " $LLSUB_RANK
echo "Number of tasks in sweep: " $LLSUB_SIZE

# Set output location.
BASE_WORKDIR="/home/gridsan/adaigavane/ActionAngleNetworks/workdirs"

# Start sweep.
python -m sweep_main --index=$LLSUB_RANK --sweep_file=$SWEEP --config=$CONFIG --base_workdir=$BASE_WORKDIR
