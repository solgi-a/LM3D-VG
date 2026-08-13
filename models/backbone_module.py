import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

try:
    from pointnet2_ops.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
except ImportError:
    # ABLATION: Python fallback selected by ABLATION.POINTNET_IMPL (default: the
    # optimized pointnet_py_adv). The CUDA extension above still wins when installed.
    from experiments.ablation.ablation_config import ABLATION
    if ABLATION.POINTNET_IMPL == "pointnet2_python":
        from pointnet2_python.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
    else:
        from pointnet_py_adv.pn import PointnetSAModuleVotes, PointnetFPModule

class Pointnet2Backbone(nn.Module):
    r"""Pointnet++ single-scale grouping backbone. input_feature_dim e.g. 3 for RGB."""
    def __init__(self, args, input_feature_dim=0):
        super().__init__()

        self.input_feature_dim = input_feature_dim

        if args.use_multiview:
            num_param1 = 64
            num_param2 = 128

        else:
            num_param1 = 64
            num_param2 = 128


        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, num_param1, num_param1, num_param2],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[num_param2, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., :3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, data_dict):
        r"""Point cloud is (B, N, 3 + input_feature_dim), formatted as (x, y, z, features...)."""
        
        pointcloud = data_dict["point_clouds"]

        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict['sa1_inds'] = fps_inds
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features


        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        data_dict['sa2_inds'] = fps_inds
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        data_dict['sa3_xyz'] = xyz
        data_dict['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        data_dict['sa4_xyz'] = xyz
        data_dict['sa4_features'] = features

        features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'], data_dict['sa3_features'], data_dict['sa4_features'])
        features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'], data_dict['sa2_features'], features)
        data_dict['fp2_features'] = features
        data_dict['fp2_xyz'] = data_dict['sa2_xyz']
        num_seed = data_dict['fp2_xyz'].shape[1]
        data_dict['fp2_inds'] = data_dict['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return data_dict

if __name__=='__main__':
    backbone_net = Pointnet2Backbone(input_feature_dim=3).cuda()
    print(backbone_net)
    backbone_net.eval()
    out = backbone_net(torch.rand(16,20000,6).cuda())
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
