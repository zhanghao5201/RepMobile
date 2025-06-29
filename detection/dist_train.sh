#!/usr/bin/env bash
source ~/.zhshrc_net
conda activate zhpose

cd ~/zhanghao5201/RepMobile/detection
pwd

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
GPUS=$4
PORT=$5
RESUME=$6
PYTHONPATH="~/zhanghao5201/RepMobile/detection/src":$PYTHONPATH 
srun --partition=$PARTITION --quotatype=spot --async \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
    train.py $CONFIG --resume-from=$RESUME --launcher pytorch --auto-resume
# --async 
# sh dist_train.sh gvembodied coct2 configs/mask_rcnn_repmobile_small_fpn_1x_coco.py 8 20234 'work_dirs/mask_rcnn_repmobile_small_fpn_1x_coco/latest.pth'
# sh dist_train.sh gvembodied coct1 configs/mask_rcnn_repmobile_large_fpn_1x_coco.py 8 20236 'work_dirs/mask_rcnn_repmobile_large_fpn_1x_coco/latest.pth'
# sh dist_train.sh gvembodied coct3 configs/mask_rcnn_repmobile_large_fpn_1x_coco_v2.py 8 20237 'work_dirs/mask_rcnn_repmobile_large_fpn_1x_coco_v2/latest.pth'
