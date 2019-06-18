#!/bin/sh
### –- specify queue --
#BSUB -q gpuv100

### -- set the job Name --
#BSUB -J elmoMLP3

### -- ask for number of cores (default: 1) --
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
#BSUB -R "select[gpu32gb]"
# request 8GB of system-memory
#BSUB -R "rusage[mem=20GB]"

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o hpc-output/job-%J.out
#BSUB -e hpc-output/job_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load python3
module load cuda/9.1
export PYTHONPATH=
source ~/stdpy3/bin/activate

python3 main.py secpred --model elmo_MLP3 --trainset data/SecPred/train_raw.npz --testset data/SecPred/test_raw.npz --batch_size 128 --epochs 400 --learning_rate 1e-3 --clip 1 --optimizer adam --input_size 2560 --n_l1 1000 --n_l2 1000 --n_rnn_hid 500 --cb513 --raw --do_testing
