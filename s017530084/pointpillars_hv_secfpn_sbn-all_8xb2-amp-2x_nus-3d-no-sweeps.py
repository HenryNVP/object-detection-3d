_base_ = [
    '../../mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py',
]

# Remove sweeps from test pipeline to work without sweep files
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 4],  # x, y, z, intensity (skip time dimension to match model expectations)
        backend_args=None),
    # Removed LoadPointsFromMultiSweeps - no sweep files available
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', 
                point_cloud_range=[-50, -50, -5, 50, 50, 3])
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Also update eval_pipeline
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 4],  # x, y, z, intensity (skip time dimension to match model expectations)
        backend_args=None),
    # Removed LoadPointsFromMultiSweeps - no sweep files available
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Update test_dataloader and val_dataloader to use modified pipelines
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

