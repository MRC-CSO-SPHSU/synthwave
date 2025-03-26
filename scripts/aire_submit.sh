#!/bin/bash

# Set current time for directory naming
TIME=`date +%Y_%m_%d_%H_%M_%S`

# Create these if they don't exist
mkdir -p "$2"/logs
mkdir -p "$2"/logs/logs
mkdir -p "$2"/logs/errors


sbatch scripts/aire_run.sh -d "$2" -f "$4" -e "$6" -t $TIME
#bash scripts/aire_run.sh -d "$2" -f "$4" -e "$6" -t $TIME  # Testing

# If no errors...
exit 0
