# End-to-end training for UniBEV_CNW
# CNW
# Encoder Dimension: 256
# Decoder Dimension: 256

_base_ = ['unibev_nus_LC_cnw_256_modality_dropout_freeze_unibev_train_auxiliary_unetfeaturemapping3layers.py']

eval_interval = 10000  # Reduced to every 2 epochs to save memory
val_interval = 1
samples_per_gpu = 2
workers_per_gpu = 4  # Reduced to free memory
max_epochs = 36
save_interval = 2
log_interval = 1
fusion_method = 'linear'
feature_norm = 'ChannelNormWeights'
modality_dropout_prob = 0.5

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
sub_dir = 'mmdet3d_bevformer/'
train_ann_file = sub_dir + 'nuscenes_annotation_files_custom/sampled_quarter_nuscenes_infos_temporal_train.pkl'
val_ann_file = sub_dir + 'nuscenes_infos_temporal_val.pkl'
work_dir = './outputs/train/unibev_nus_LC_cnw_256_modality_dropout_freeze_unibev_train_auxiliary_unetfeaturemapping3layers_quarter_nuscenes'

load_from = '/home/mingdayang/UniBEV/projects/UniBEV/checkpoints/unibev_cnw_256_nus_MD.pth'

resume_from = None
plugin = True
plugin_dir = 'mmdet3d/unibev_plugin/'

## nuscenes and pointpillars setting
point_cloud_range = [-54, -54, -5, 54, 54, 3]
voxel_size = [0.075, 0.075, 0.2]
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
file_client_args = dict(backend='disk')
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

### model settings
img_scale = (1600, 900)
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
if fusion_method == 'linear':
    dec_scale_factor = 1
elif fusion_method == 'avg':
    dec_scale_factor = 1
elif fusion_method == 'cat':
    dec_scale_factor = 2

_encoder_layers_ = 3
_num_levels_ = 1
_num_points_in_pillar_cam_ = 4
_num_points_in_pillar_lidar_ = 4
bev_h_ = 200
bev_w_ = 200
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

runner = dict(type='EpochBasedRunner',
              max_epochs=max_epochs)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=5,  # Reduced from 10 to save memory
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),

    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names), ## which DefaultFormat
    dict(type='CustomCollect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']) ## which data collection
    # dict(type='CustomCollect3D', keys=['points', 'img']) ## which data collection
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['points', 'img'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + train_ann_file,
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + val_ann_file,
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + val_ann_file,
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))


# evaluation = dict(interval=eval_interval, pipeline=test_pipeline)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
# max_norm=10 is better for SECOND
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

# runtime settings
total_epochs = max_epochs

checkpoint_config = dict(interval=save_interval)
log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(type='CustomWandbLoggerHook',
            by_epoch=True,
            with_step=True,
            ignore_last=True,
            interval=1,
            log_artifact=True,
            # out_suffix=('.log.json', '.log', '.py', 'pth', 'pt'),
            # log_checkpoint=True, # Doens't work.
            init_kwargs=dict(
                project='Feature Mapping UniBEV',
                notes='Training UniBEV with feature mapping 3 layers, quarter nuscenes, freeze unibev weights, auxiliary loss',
                allow_val_change=True,
                save_code=True,
                config=dict(
                    model='unet_feature_mapping_3layers',
                    work_dir=work_dir,
                    total_epochs=max_epochs,
                    batch_size=samples_per_gpu,
                    fusion_method=fusion_method,
                    feature_norm=feature_norm,
                    modality_dropout_prob=modality_dropout_prob,
                    optimizer=optimizer,
                    lr_config=lr_config,
                ),
            ))
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
custom_hooks = [
    dict(type='CheckpointLateStageHook',
         start=21,
         priority=60),

]
workflow = [('train', 1), ('val', 1)]  
