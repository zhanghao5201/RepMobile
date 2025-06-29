#!/usr/bin/env bash
source ~/.zhshrc_net
conda activate RepMobile

cd ~/zhanghao5201/RepMobile/detection
pwd

PARTITION=$1
JOB_NAME=$2
PORT=$3
CONFIG=$4
CHECKPOINT=$5
GPUS=$6
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="~/zhanghao5201/RepMobile/detection/src":$PYTHONPATH 
srun --partition=$PARTITION --quotatype=reserved \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python test.py \
    $CONFIG \
    $CHECKPOINT \
    --eval bbox segm --work-dir work_dirs/mask_rcnn_repvit_m1_1_fpn_1x_coco

# sh dist_test.sh XXX coct1 20345 configs/mask_rcnn_repmobile_large_fpn_1x_coco.py 'work_dirs/mask_rcnn_repmobile_large_fpn_1x_coco/epoch_12.pth' 1  
