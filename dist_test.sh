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
RESUME=$6

PYTHONPATH="~/zhanghao5201/RepMobile/src":$PYTHONPATH 
srun --partition=$PARTITION --quotatype=spot \
--mpi=pmi2 \
--gres=gpu:$GPUS \
--job-name=${JOB_NAME} \
--kill-on-bad-exit=1 \
python -u -m torch.distributed.launch --nproc_per_node=$GPUS --master_port $PORT --use_env main.py --model $MODELNAME --data-path /mnt/petrelfs/share/imagenet/images --output_dir "checkpoints/${MODELNAME}_300d" --eval --dist-eval --resume $RESUME

# --async 

#sh dist_test.sh XXX RepMobile_M 23426 4 RepMobile_M pretrain_model/RepMobile_M.pth Acc@1 78.782
#sh dist_test.sh XXX RepMobile_S 23426 4 RepMobile_S pretrain_model/RepMobile_S.pth Acc@1 77.232 
#sh dist_test.sh XXX RepMobile_L 23426 4 RepMobile_L pretrain_model/RepMobile_L.pth Acc@1 80.866