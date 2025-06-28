#!/usr/bin/env bash
source ~/.zhshrc_net
conda activate RepMobile

cd ~/zhanghao5201/RepMobile/segmentation
pwd

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=$4
PORT=$5
RESUME=$6
PYTHONPATH="~/zhanghao5201/RepMobile/segmentation/src":$PYTHONPATH 
srun --partition=$PARTITION --quotatype=spot --async \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
    tools/train.py $CONFIG --resume-from=$RESUME --launcher pytorch 


# sh tools/dist_train.sh gvembodied repseg3 configs/sem_fpn/fpn_repmobile_large_ade20k_40k.py 8 20233 work_dirs/fpn_repmobile_large_ade20k_40k/latest.pth