_base_ = [
    './fcos_r50-caffe_fpn_gn-head_1x_coco.py',
]

num_classes = 18

# model settings
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
    # добавила наше количество классов
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # testing settings
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

METAINFO = {
        'classes':
        ('minecraft-mobs', 'bee', 'chicken', 'cow', 'creeper', 'enderman', 'fox',
         'frog', 'ghast', 'goat', 'llama', 'pig', 'sheep', 'skeleton', 'spider',
         'turtle', 'wolf', 'zombie'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255)
         ]
    }

#----------- TRAIN_PIPELINE---------
# За основу взят код из документации mmdetection
train_pipeline = [  # Training data processing pipeline
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='LoadAnnotations',  # Second pipeline to load annotations for current image
        with_bbox=True),  # Whether to use bounding box, True for detection
    dict(type='Resize',  # Pipeline that resizes the images and their annotations
        scale=(512, 512),  # The largest scale of the images
        keep_ratio=True  # Whether to keep the ratio between height and width
        ),
    dict(type='RandomFlip', # Augmentation pipeline that flips the images and their annotations
        prob=0.5), # The probability to flip
    dict(type='PackDetInputs')
]

#-----------TEST_PIPELINE-------------
# За основу взят код из документации mmdetection
test_pipeline = [  # Testing data processing pipeline
    dict(type='LoadImageFromFile'),# First pipeline to load images from file path
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),  # Pipeline that resize the images
    dict(type='PackDetInputs',  # Pipeline that formats the annotation data and decides which keys in the data should be packed into data_samples
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

#------Датасет------
data_root = 'datasets/minecraft/'
train_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type='CocoDataset', data_root=data_root,
        ann_file='annotations/train/annotations.json',
        data_prefix=dict(img='train/'),
        metainfo=METAINFO, pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type='CocoDataset', data_root=data_root,
        ann_file='annotations/valid/annotations.json',
        data_prefix=dict(img='valid/'),
        metainfo=METAINFO, pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=2, num_workers=2,
    dataset=dict(
        type='CocoDataset', data_root=data_root,
        ann_file='annotations/test/annotations.json',
        data_prefix=dict(img='test/'),
        metainfo=METAINFO, pipeline=test_pipeline))

val_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'annotations/valid/annotations.json',
                     metric='bbox')
test_evaluator = dict(type='CocoMetric',
                     ann_file=data_root + 'annotations/test/annotations.json',
                     metric='bbox')


#------Интервал сохранения каждую эпоху-----
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1), # Save checkpoints periodically
)

#------MAX_EPOCHS------
train_cfg = dict(
    max_epochs=12,  # Maximum training epochs
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type


# learning rate
param_scheduler = [
    dict(type='ConstantLR', factor=1.0 / 3, by_epoch=False, begin=0, end=300),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

#-----------OPTIMAIZER--------
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='SGD',lr=0.001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

#----------RUNNER---------
# runner = dict(type='EpochBasedRunner', max_epochs=12)

# --- Обучение ---
# train_cfg = dict(max_epochs=12, val_interval=1)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')



