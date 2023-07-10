#!/bin/bash
### Job name
#BSUB -J nerf-experiments


### Job queue
#BSUB -q gpua100 

### GPUs to request and if to reserve it exclusively

#BSUB -gpu "num=1:mode=exclusive_process"

### Cores to request
#BSUB -n 8

### Force cores to be on same host
#BSUB -R "span[hosts=1]" 

### Number of threads for OpenMP parallel regions
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC


### Amount of memory to request
#BSUB -R "rusage[mem=32GB]"


### Wall time (HH:MM), how long before killing task
#BSUB -W 24:00


### Output and error file. %J is the job-id -- 
### -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo .sprinkle/log/%J-nerf-experiments.txt
#BSUB -eo .sprinkle/error/%J-nerf-experiments.txt


# Get shell environment
source ~/.bashrc


# Change working directory
cd /zhome/56/e/155505/Desktop/bachelor/nerf-experiments/original-nerf-sarphiv


# Activate environment for script
conda activate nerf-experiments-env


# If unable to activate environment, inform and exit
if [[ $? -ne 0 ]]; then
    echo "Failed to activate environment (nerf-experiments-env) for job ($LSB_JOBID)." >&2
    echo 'Please run sprinkle setup before submitting the job.' >&2
    exit 1
fi


# Run job script and save output to file
# NOTE: %J is not available so using environment variable
python main.py  > ../.sprinkle/output/$LSB_JOBID-nerf-experiments.txt

