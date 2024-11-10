# inference/test configuration for bevformer_pp_fusion trained on the complete nuscenes dataset
# inference with only LiDAR input
# inference on complete NuScenes dataset

_base_ = ['../unibev_nus_LC_cnw_256_modality_dropout.py']

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
sub_dir = 'mmdet3d_bevformer/'
val_ann_file = sub_dir + 'mini_nuscenes_infos_temporal_val.pkl'
file_client_args = dict(backend='disk')
bev_h_ = 200
bev_w_ = 200
dist_params = dict(backend='gloo')

outdir = 'outputs/inference/baseline/unibev_val_L_full_mini_nuscenes'
keys = ['fused_bev_embed', 'img_mlvl_feats', 'img_bev_embed', 'pts_mlvl_feats', 'pts_bev_embed', 'query', 'bev_queries', 'bev_pos', 'query_pos', 'reference_points']
special_keys = []
attrs = []

input_modality =  dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (1600, 900)
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

model = dict(
    use_lidar=input_modality['use_lidar'],
    use_radar=input_modality['use_radar'],
    use_camera=input_modality['use_camera'],
    pts_bbox_head=dict(
        transformer=dict(
            vis_output=dict(
                outdir= outdir,
                keys=keys,
                special_keys=special_keys,
                attrs=attrs
            )
        )
    )
)

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
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['points'])
        ])
]

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + val_ann_file,
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1,
        test_mode=True,
        box_type_3d='LiDAR'),
)

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
evaluation = dict(pipeline=test_pipeline)
