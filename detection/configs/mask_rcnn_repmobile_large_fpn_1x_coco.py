_base_ = [
    '_base_/models/mask_rcnn_r50_fpn.py',
    '_base_/datasets/coco_instance.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    pretrained=None,
    backbone=dict(
        type='RepMobile_L',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='~/zhanghao5201/RepMobile/pretrain_model/RepMobile_L.pth',
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[96,192,384,768],
        out_channels=256,
        num_outs=5,norm_cfg=dict(type='SyncBN', requires_grad=True)),
    roi_head=dict(
       bbox_head=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
       mask_head=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
       )    
    )
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002,  weight_decay=0.05,betas=(0.9, 0.999))  # 0.0001
optimizer_config = dict(grad_clip=None)