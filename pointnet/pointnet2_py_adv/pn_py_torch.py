
import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

_P2P_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pointnet2_python"
)
if _P2P_DIR not in sys.path:
    sys.path.append(_P2P_DIR)
import pt_utils  # noqa: E402  (pointnet2_python/pt_utils.py)

_CHUNK_ELEMS = 16_777_216


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def furthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    with torch.no_grad():
        distance = torch.full((B, N), 1e10, dtype=xyz.dtype, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz.gather(1, farthest.view(B, 1, 1).expand(B, 1, 3))
            dist = torch.sum((xyz - centroid) ** 2, -1)
            torch.minimum(distance, dist, out=distance)
            farthest = torch.max(distance, -1)[1]
    return centroids


def gather_operation(points, idx):
    B, N, C = points.shape
    flat = idx.reshape(B, -1)
    out = torch.gather(points, 1, flat.unsqueeze(-1).expand(B, flat.shape[1], C))
    return out.reshape(*idx.shape, C)


def _gather_channels(features, idx):
    B, C, N = features.shape
    _, S, K = idx.shape
    flat = idx.reshape(B, 1, S * K).expand(B, C, S * K)
    return torch.gather(features, 2, flat).reshape(B, C, S, K)


def ball_query(radius, nsample, xyz, new_xyz):
    with torch.no_grad():
        B, N, _ = xyz.shape
        _, S, _ = new_xyz.shape
        device = xyz.device
        r2 = radius ** 2

        arange = torch.arange(N, dtype=torch.int32, device=device).view(1, 1, N)
        sentinel = torch.tensor(N, dtype=torch.int32, device=device)

        chunk = max(1, min(S, _CHUNK_ELEMS // max(B * N, 1)))
        out = torch.empty(B, S, nsample, dtype=torch.int64, device=device)

        for s0 in range(0, S, chunk):
            s1 = min(s0 + chunk, S)
            sqrdists = square_distance(new_xyz[:, s0:s1], xyz)
            cand = torch.where(sqrdists <= r2, arange, sentinel)
            idx = torch.topk(cand, nsample, dim=-1, largest=False, sorted=True)[0]
            first = idx[:, :, 0:1].expand_as(idx)
            idx = torch.where(idx == sentinel, first, idx)
            out[:, s0:s1] = idx.long()
    return out


class QueryAndGroup(nn.Module):

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False,
                 normalize_xyz=False, sample_uniformly=False, ret_unique_cnt=False):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]), device=xyz.device)
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique,
                                               (self.nsample - num_unique,),
                                               dtype=torch.long, device=xyz.device)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_t = xyz.transpose(1, 2).contiguous()
        grouped_xyz = _gather_channels(xyz_t, idx)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz = grouped_xyz / self.radius

        if features is not None:
            grouped_features = _gather_channels(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, \
                "It's not possible to have no features and no xyz coordinates to use as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        return ret[0] if len(ret) == 1 else tuple(ret)


class GroupAll(nn.Module):

    def __init__(self, use_xyz=True, ret_grouped_xyz=False):
        super().__init__()
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz

    def forward(self, xyz, features=None):
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz
        if self.ret_grouped_xyz:
            return new_features, grouped_xyz
        return new_features


def three_nn(unknown, known):
    with torch.no_grad():
        sqrdists = square_distance(unknown, known)
        dist, idx = torch.topk(sqrdists, 3, dim=-1, largest=False, sorted=True)
    return dist, idx


def three_interpolate(features, idx, weight):
    B, C, M = features.shape
    _, n, _ = idx.shape
    gathered = _gather_channels(features, idx)
    return torch.sum(gathered * weight.view(B, 1, n, 3), dim=-1)


class PointnetSAModuleVotes(nn.Module):

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
            self.sigma = self.radius / 2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt

        if npoint is not None:
            self.grouper = QueryAndGroup(
                radius, nsample, use_xyz=use_xyz, ret_grouped_xyz=True,
                normalize_xyz=normalize_xyz, sample_uniformly=sample_uniformly,
                ret_unique_cnt=ret_unique_cnt)
        else:
            self.grouper = GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec) > 0:
            mlp_spec[0] += 3
        self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None,
                inds: torch.Tensor = None):
        if inds is None:
            inds = furthest_point_sample(xyz, self.npoint)
        else:
            assert inds.shape[1] == self.npoint
        new_xyz = gather_operation(xyz, inds) if self.npoint is not None else None

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(xyz, new_xyz, features)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(xyz, new_xyz, features)

        new_features = self.mlp_module(grouped_features)

        if self.pooling == 'max':
            new_features = F.max_pool2d(new_features,
                                        kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'avg':
            new_features = F.avg_pool2d(new_features,
                                        kernel_size=[1, new_features.size(3)])
        elif self.pooling == 'rbf':
            rbf = torch.exp(-1 * grouped_xyz.pow(2).sum(1, keepdim=False)
                            / (self.sigma ** 2) / 2)
            new_features = torch.sum(new_features * rbf.unsqueeze(1), -1,
                                     keepdim=True) / float(self.nsample)

        new_features = new_features.squeeze(-1)

        if not self.ret_unique_cnt:
            return new_xyz, new_features, inds
        return new_xyz, new_features, inds, unique_cnt


class PointnetFPModule(nn.Module):

    def __init__(self, *, mlp: List[int], bn: bool = True):
        super().__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown: torch.Tensor, known: torch.Tensor,
                unknown_feats: torch.Tensor, known_feats: torch.Tensor) -> torch.Tensor:
        if known is not None:
            dist, idx = three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_feats = three_interpolate(known_feats, idx, weight)
        else:
            interpolated_feats = known_feats.expand(*known_feats.size()[0:2],
                                                    unknown.size(1))

        if unknown_feats is not None:
            new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)
        return new_features.squeeze(-1)
