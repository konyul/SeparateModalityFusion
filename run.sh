## lidar and camera
work_dir=work_dirs/masking_strategy/version2/concat_smt
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking.py 4 --work-dir $work_dir
for failures in 'beam_reduction' 'camera_view_drop' 'limited_fov' 'object_failure' 'spatial_misalignment' 'lidar_drop' 'occlusion'
do
    bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_img_mask/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
done

##

## lidar sota
# work_dir=work_dirs/masking_strategy/version1/sota
# bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10.py 4 --work-dir $work_dir
# for failures in 'beam_reduction' 'camera_view_drop' 'limited_fov' 'object_failure' 'spatial_misalignment' 'lidar_drop' 'occlusion'
# do
#     bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
# done
##



## camera

# work_dir=work_dirs/masking_strategy/version3/camera
# bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_img_mask.py 4 --work-dir $work_dir
# for failures in 'beam_reduction' 'camera_view_drop' 'limited_fov' 'object_failure' 'spatial_misalignment' 'lidar_drop' 'occlusion'
# do
#     bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_img_mask/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
# done

##
