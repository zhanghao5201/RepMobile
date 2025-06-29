_base_ = [
    '../default_runtime.py',
]

# dataset settings
dataset_type = 'AstronomyDataset'
data_root = '/mnt/petrelfs/zhanghao1/network_design/AI4Astronomy/data/AI4Astronomy/'
image_size = (150, 150)

backend_args = None
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(150,150), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(150,150),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/Astronomy_instances_train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/Astronomy_instances_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/Astronomy_instances_val.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))

# train_dataloader = dict(
#     batch_size=64,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/Astronomy_instances_train.json',
#         data_prefix=dict(img=''),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=train_pipeline))

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='annotations/Astronomy_instances_val.json',
#         data_prefix=dict(img=''),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_dataloader = val_dataloader
evaluation = dict(metric=['bbox', 'segm'])

# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + 'annotations/Astronomy_instances_val.json',
#     metric=['bbox', 'segm'],
#     format_only=False)
# test_evaluator = val_evaluator


optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    )
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
max_iters = 80200  #ys:184375   118287
interval = 1000
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=250),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        # 88 ep = [163889 iters * 64 images/iter / 118000 images/ep
        # 96 ep = [177546 iters * 64 images/iter / 118000 images/ep
        # milestones=[163889, 177546], #ys
        milestones=[71200, 77200],
        gamma=0.1)
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=interval,
    dynamic_intervals=dynamic_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        interval=interval,
        max_keep_ckpts=5, save_best='coco/bbox_mAP'))
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

auto_scale_lr = dict(base_batch_size=64)
