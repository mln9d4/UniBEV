# inference/test configuration for bevformer_pp_fusion trained on the complete nuscenes dataset
# inference with only LiDAR input
# inference on complete NuScenes dataset

_base_ = ['../unibev_nus_LC_cnw_256_modality_dropout.py']

# outdir = 'outputs/inference/showcasing_worsened_prediction/beams_missing/mmdet_module_reduction/unibev_val_LC_full_cnw_256_mini_nuscenes_L_8beams_C_Good'
# keys = ['fused_bev_embed', 'img_mlvl_feats', 'img_bev_embed', 'pts_mlvl_feats', 'pts_bev_embed', 'query', 'bev_queries', 'bev_pos', 'query_pos', 'reference_points']
# special_keys = []
# attrs = []



dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
sub_dir = 'mmdet3d_bevformer/'
val_ann_file = sub_dir + 'mini_nuscenes_infos_temporal_val.pkl'
file_client_args = dict(backend='disk')
bev_h_ = 200
bev_w_ = 200
dist_params = dict(backend='gloo')

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
input_modality =  dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (1600, 900)
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

num_beam_to_drop = 8
num_beam_sensor = 32

model = dict(
    use_lidar=input_modality['use_lidar'],
    # use_radar=input_modality['use_radar'],
    use_camera=input_modality['use_camera'],
    # pts_bbox_head=dict(
    #     transformer=dict(
    #         vis_output=dict(
    #             outdir= outdir,
    #             keys=keys,
    #             special_keys=special_keys,
    #             attrs=attrs
    #         )
    #     )
    # )
)

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5
    ),
    # dict(
    #     type='RemoveLiDARBeamsSpaced',
    #     num_beam_to_drop=num_beam_to_drop,
    #     num_beam_sensor=num_beam_sensor,
    #     coord_type='LIDAR',
    #     save_fig=False
    # ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True
    ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     use_dim=[0, 1, 2, 3, 4],
    #     file_client_args=file_client_args,
    #     pad_empty_sweeps=True,
    #     remove_close=True
    # ),

    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='DownscaleUpscale', 
        save_fig=True,
        target_width=640,
        target_height=360),
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
        type='RemoveLiDARBeamsSpaced',
        num_beam_to_drop=16,
        num_beam_sensor=32,
        coord_type='LIDAR',
        save_fig=False),
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