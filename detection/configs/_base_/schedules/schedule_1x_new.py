# optimizer
optimizer = dict(type='SGD', lr=0.02, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,  # 0.001
    )
runner = dict(type='EpochBasedRunner', max_epochs=12)
