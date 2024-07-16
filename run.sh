work_dir=work_dirs/masking_strategy/version2/e2e_robusthead
bash tools/dist_train.sh projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_e2e.py 4 --work-dir $work_dir 
# for failures in 'lidar_drop' 'camera_view_drop'  'limited_fov' 'beam_reduction' 'object_failure'  'occlusion'
# do
#    bash tools/dist_test.sh projects/BEVFusion/configs/e2e/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_$failures.py $work_dir/epoch_5.pth 4 --work-dir $work_dir/$failures
# done