import torch
import copy
aa = torch.load("bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af_.pth")
bb = copy.deepcopy(aa)
for key in aa['state_dict'].keys():
    if len(aa['state_dict'][key].shape) == 5 and aa['state_dict'][key].shape[3] == 3:
        bb['state_dict'][key] = aa['state_dict'][key].permute(1,2,3,4,0)
torch.save(bb,"bevfusion_mmdetection3d_permute.pth")
import pdb;pdb.set_trace()