_base_ = [
    '../bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_e2e.py'
]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

beam_reduction=True
spatial_misalignment=False
lidar_stuck=False
camera_stuck=False
limited_fov=False
object_failure=False
camera_view_drop=False
if camera_view_drop==True:
    mean=[0,0,0]
    std=[1,1,1]
else:
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]  
test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        camera_view_drop=camera_view_drop),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args,
        reduce_beams=beam_reduction),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args,
        reduce_beams=beam_reduction,
        limited_fov=limited_fov),
    dict(type='Randomdropforeground',
        object_failure=object_failure),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

if object_failure:
    test_pipeline.insert(3,
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=False,
        with_attr_label=False))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, modality=input_modality, spatial_misalignment=spatial_misalignment, lidar_stuck=lidar_stuck, camera_stuck=camera_stuck, object_failure=object_failure))
test_dataloader = val_dataloader

val_cfg = dict()
test_cfg = dict()
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
del _base_.custom_hooks
