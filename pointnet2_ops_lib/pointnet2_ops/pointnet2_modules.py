# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Pointnet2 layers, extended from https://github.com/erikwijmans/Pointnet2_PyTorch
with uniform per-region sampling and returned sample indices (for votenet). '''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from typing import List

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pointnet2_utils
import pt_utils

class PointnetSAModuleVotes(nn.Module):
    """ Set abstraction module (_PointnetSAModuleBase/PointnetSAModuleMSG) that also
    returns point indices, for GT vote supervision. """

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None,
            normalize_xyz: bool = False, 
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)


    def forward(self, xyz: torch.Tensor,features: torch.Tensor = None,inds: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:

        """
        xyz: (B,N,3); features: (B,C,N); inds: (B,npoint) indices into xyz (optional).
        Returns new_xyz (B,npoint,3), new_features (B,sum(mlps[-1]),npoint), inds.
        """

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert(inds.shape[1] == self.npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, inds).transpose(1, 2).contiguous() if self.npoint is not None else None

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)                # (B, C, npoint, nsample), (B,3,npoint,nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(xyz, new_xyz, features)    # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        new_features = self.mlp_module(grouped_features)                                        # (B, mlp[-1], npoint, nsample)

        if self.pooling == 'max':
            
            new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])    # (B, mlp[-1], npoint, 1)

        elif self.pooling == 'avg':

            new_features = F.avg_pool2d(new_features, kernel_size=[1, new_features.size(3)])    # (B, mlp[-1], npoint, 1)

        elif self.pooling == 'rbf': 
            
            # RBF-weighted sum of features, normalized by nsample and sigma
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1,keepdim=False) / (self.sigma**2) / 2)                 # (B, npoint, nsample)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1, keepdim=True) / float(self.nsample)   # (B, mlp[-1], npoint, 1)
            
        new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        else:
            return new_xyz, new_features, inds, unique_cnt

class PointnetFPModule(nn.Module):

    """ Propagates features of one set to another. """

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor,unknown_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        
        """
        unknown/known: (B,n/m,3) xyz; unknown_feats: (B,C1,n); known_feats: (B,C2,m) to propagate.
        Returns new_features (B, mlp[-1], n).
        """

        if known is not None:

            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)

        else: 
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2], unknown.size(1))
        
        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats],dim=1)                 #(B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
