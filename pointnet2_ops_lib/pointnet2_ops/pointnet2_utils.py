# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''

import torch
import torch.nn as nn
import warnings
import sys
import os
from torch.autograd import Function
from typing import *

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp

    warnings.warn("Unable to load pointnet2_ops cpp extension. JIT Compiling.")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r""" Iterative furthest point sampling: picks npoint features from xyz (B,N,3), N>npoint,
        maximizing minimum distance. Returns (B, npoint) indices. """
        out = _ext.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r""" Gathers features (B,C,N) at idx (B,npoint) -> (B,C,npoint). """

        ctx.save_for_backward(idx, features)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r""" 3 nearest neighbors of unknown (B,n,3) in known (B,m,3).
        Returns dist (B,n,3) l2 distances and idx (B,n,3) neighbor indices. """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)

        ctx.mark_non_differentiable(dist, idx)

        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        return ()


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r""" Weighted linear interpolation of features (B,c,m) using idx/weight (B,n,3)
        neighbors -> (B,c,n). """
        ctx.save_for_backward(idx, weight, features)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r""" grad_out (B,c,n) -> grad_features (B,c,m), None, None. """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r""" Groups features (B,C,N) at idx (B,npoint,nsample) -> (B,C,npoint,nsample). """
        ctx.save_for_backward(idx, features)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r""" grad_out (B,C,npoint,nsample) -> grad_features (B,C,N), None. """
        idx, features = ctx.saved_tensors
        N = features.size(2)

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, torch.zeros_like(idx)


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r""" Ball query: for each of new_xyz (B,npoint,3), up to nsample neighbors of xyz (B,N,3)
        within radius. Returns (B,npoint,nsample) indices. """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)

        ctx.mark_non_differentiable(output)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        return ()


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r""" Groups features via a ball query of given radius/nsample. """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        # type: (QueryAndGroup, float, int, bool, bool, bool, bool, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert(self.sample_uniformly)

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r""" xyz (B,N,3), new_xyz centroids (B,npoint,3), features (B,C,N) -> (B,3+C,npoint,nsample). """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
        
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)            # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)        # center on ball centroid
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

class GroupAll(nn.Module):
    r""" Groups all features into a single group. """

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        # type: (GroupAll, bool, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
        r""" xyz (B,N,3), new_xyz ignored, features (B,C,N) -> (B,C+3,1,N). """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        else:
            return new_features
