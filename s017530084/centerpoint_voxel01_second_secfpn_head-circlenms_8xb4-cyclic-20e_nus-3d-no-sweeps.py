_base_ = [
    '../../mmdetection3d/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py',
]

# Remove sweeps from test pipeline to work without sweep files
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],  # x, y, z, time, intensity (CenterPoint expects 5 dims)
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
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Also update eval_pipeline
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],  # x, y, z, time, intensity (CenterPoint expects 5 dims)
        backend_args=None),
    # Removed LoadPointsFromMultiSweeps - no sweep files available
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Update test_dataloader and val_dataloader to use modified pipelines
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

