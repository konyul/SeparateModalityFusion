# lidar and camera
work_dir=work_dirs/masking_strategy/version2/x_maskfusion_bevquery
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking.py 4 --work-dir $work_dir
for failures in 'camera_view_drop' 'occlusion' 'object_failure' 'lidar_drop' 'beam_reduction'  'spatial_misalignment' 'limited_fov'
do
bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_img_mask/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
done
work_dir=work_dirs/masking_strategy/version2/x_maskfusion_bevquery_smt_longer_epoch_fix
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking_smt.py 4 --work-dir $work_dir
for failures in 'beam_reduction' 'limited_fov' 'object_failure' 'camera_view_drop' 
do
  bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_img_mask/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_9.pth 4 --work-dir $work_dir/$failures
done