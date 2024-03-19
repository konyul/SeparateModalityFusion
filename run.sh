work_dir=work_dirs/masking_strategy/version1/mask_l_defaultm_deform_attn_fg_bg_mask_patch_5_10_baseline
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10.py 4 --work-dir $work_dir
for failures in 'beam_reduction' 'limited_fov' 'object_failure' 'lidar_drop'
do
    bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
done
