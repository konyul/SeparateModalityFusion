# work_dir=work_dirs/masking_strategy/version2/concat_image_head
# bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking.py 4 --work-dir $work_dir
# for failures in 'camera_view_drop' 'lidar_drop'  'limited_fov' 'object_failure' 'spatial_misalignment'  'occlusion' 'beam_reduction'
# do
#    bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_with_imghead/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
# done

work_dir=work_dirs/masking_strategy/version2/concat_image_head_smt_img_backbone_ep18
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_deform_fg_bg_mask_patch_5_10_concat_img_masking_smt.py 4 --work-dir $work_dir
# for failures in  'camera_view_drop'  'limited_fov' 'object_failure' 'spatial_misalignment'  'occlusion' 'beam_reduction'
# do
#    bash tools/dist_test.sh projects/BEVFusion/configs/deform_failure_with_imghead/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_4.pth 4 --work-dir $work_dir/$failures/epoch_4
# done
