#!/bin/sh
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J testjob

### -- ask for number of cores (default: 1) --
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 0:30

# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "select[sxm2]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu-%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load python3
module load cuda/9.1
export PYTHONPATH=
source ~/stdpy3/bin/activate

python3 main.py > output.out

