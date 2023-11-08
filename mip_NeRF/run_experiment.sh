#!/bin/bash
#BSUB -J experiment
#BSUB -R "rusage[mem=4GB]"
###BSUB -q gpua100
###BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]" 
#BSUB -o bsub_outputs/%J.out
#BSUB -e bsub_outputs/%J.err
#BSUB -W 24:00
# -- end of LSF options --

cd /zhome/56/e/155505/Desktop/bachelor/nerf-experiments/mip_NeRF

conda activate nerf-experiments-env

# separate fine and coarse mip models
python main.py --use_seperate_coarse_fine "1" --experiment_name "mip separate fine and coarse"

# individual variance (the original mip model)
# python main.py --experiment_name "mip axis individual variance"

# evenly distributed variance
# python main.py --mip_distribute_variance "1" --experiment_name "mip axis evenly distributed variance"

