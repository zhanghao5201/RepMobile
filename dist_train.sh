#!/usr/bin/env bash
source ~/.zhshrc_net
conda activate RepMobile

cd ~/zhanghao5201/RepMobile
pwd

PARTITION=$1
JOB_NAME=$2
PORT=$3
GPUS=$4
MODELNAME=$5
MIXUP=$6
CUTMIX=$7
PYTHONPATH="~/zhanghao5201/RepMobile/src":$PYTHONPATH 
srun --partition=$PARTITION --quotatype=spot \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT --use_env main.py --model $MODELNAME --mixup $MIXUP --cutmix $CUTMIX --data-path /mnt/petrelfs/share/imagenet/images --output_dir "${MODELNAME}_300d" --resume checkpoint.pth --model_ema false

# sh dist_train.sh XXX RepMobile_M 23426 4 RepMobile_M 0.0 0.0 
# sh dist_train.sh XXX RepMobile_S 23427 4 RepMobile_S 0.0 0.0 
# sh dist_train.sh XXX RepMobile_L 20427 4 RepMobile_L 0.8 1.0