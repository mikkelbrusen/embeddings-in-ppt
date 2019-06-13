module load python3
module load cuda/9.1
export PYTHONPATH=
source ~/stdpy3/bin/activate

python3 main.py subcel --model raw_awd_concat_hid --do_testing --epochs 3
