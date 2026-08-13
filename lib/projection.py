
import torch
from torch.autograd import Function


class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims, accuracy, cuda=True):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.accuracy = accuracy
        self.cuda = cuda

        self._compute_corner_points()

    def depth_to_skeleton(self, ux, uy, depth):
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])

    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])

    def _compute_corner_points(self):
        if self.cuda:
            corner_points = torch.ones(8, 4).cuda()
        else:
            corner_points = torch.ones(8, 4)
        
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min)
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max)

        self.corner_points = corner_points

    def compute_frustum_corners(self, camera_to_world):


        corner_coords = torch.bmm(camera_to_world.repeat(8, 1, 1), self.corner_points.unsqueeze(2))

        return corner_coords

    def compute_frustum_normals(self, corner_coords):

        normals = corner_coords.new(6, 3)

        plane_vec1 = corner_coords[3][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[1][:3] - corner_coords[0][:3]
        normals[0] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        plane_vec1 = corner_coords[2][:3] - corner_coords[1][:3]
        plane_vec2 = corner_coords[5][:3] - corner_coords[1][:3]
        normals[1] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        plane_vec1 = corner_coords[3][:3] - corner_coords[2][:3]
        plane_vec2 = corner_coords[6][:3] - corner_coords[2][:3]
        normals[2] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        plane_vec1 = corner_coords[0][:3] - corner_coords[3][:3]
        plane_vec2 = corner_coords[7][:3] - corner_coords[3][:3]
        normals[3] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        plane_vec1 = corner_coords[1][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[0][:3]
        normals[4] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        plane_vec1 = corner_coords[6][:3] - corner_coords[5][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[5][:3]
        normals[5] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        return normals

    def points_in_frustum(self, corner_coords, normals, new_pts, return_mask=False):

        point_to_plane1 = (new_pts.cuda() - corner_coords[2][:3].view(-1))
        point_to_plane2 = (new_pts.cuda() - corner_coords[4][:3].view(-1))

        masks = list()
        for k, normal in enumerate(normals):
            if k < 3:
                masks.append(torch.round(torch.mm(point_to_plane1, normal.unsqueeze(1)) * 100) / 100 < 0)
            else:
                masks.append(torch.round(torch.mm(point_to_plane2, normal.unsqueeze(1)) * 100) / 100 < 0)
        mask = torch.ones(point_to_plane1.shape[0]) > 0
        mask = mask.cuda()

        for addMask in masks:
            mask = mask * addMask.squeeze()

        if return_mask:
            return mask
        else:
            return torch.sum(mask)
            
    def points_in_frustum_cpu(self, corner_coords, normals, new_pts, return_mask=False):

        point_to_plane1 = (new_pts - corner_coords[2][:3].view(-1))
        point_to_plane2 = (new_pts - corner_coords[4][:3].view(-1))

        masks = list()
        for k, normal in enumerate(normals):
            if k < 3:
                masks.append(torch.round(torch.mm(point_to_plane1, normal.unsqueeze(1)) * 100) / 100 < 0)
            else:
                masks.append(torch.round(torch.mm(point_to_plane2, normal.unsqueeze(1)) * 100) / 100 < 0)
        mask = torch.ones(point_to_plane1.shape[0]) > 0

        for addMask in masks:
            mask = mask * addMask.squeeze()

        if return_mask:
            return mask
        else:
            return torch.sum(mask)

    def compute_projection(self, points, depth, camera_to_world):

        num_points = points.shape[0]
        world_to_camera = torch.inverse(camera_to_world)

        ind_points = torch.arange(0, num_points, out=torch.LongTensor()).cuda()
        coords = camera_to_world.new(4, num_points)
        coords[:3, :] = torch.t(points)
        coords[3, :].fill_(1)

        corner_coords = self.compute_frustum_corners(camera_to_world)
        normals = self.compute_frustum_normals(corner_coords)

        mask_frustum_bounds = self.points_in_frustum(corner_coords, normals, points, return_mask=True).cuda()

        if not mask_frustum_bounds.any():
            return None
        ind_points = ind_points[mask_frustum_bounds]
        coords = coords[:, ind_points]

        camera = torch.mm(world_to_camera, coords)

        camera[0] = (camera[0] * self.intrinsic[0][0]) / camera[2] + self.intrinsic[0][2]
        camera[1] = (camera[1] * self.intrinsic[1][1]) / camera[2] + self.intrinsic[1][2]
        image = torch.round(camera).long()

        valid_ind_mask = torch.ge(image[0], 0) * torch.ge(image[1], 0) * torch.lt(image[0], self.image_dims[0]) * torch.lt(image[1], self.image_dims[1])
        if not valid_ind_mask.any():
            return None
        valid_image_ind_x = image[0][valid_ind_mask]
        valid_image_ind_y = image[1][valid_ind_mask]
        valid_image_ind = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x

        depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind.cuda())
        depth_mask = depth_vals.ge(self.depth_min) * depth_vals.le(self.depth_max) * torch.abs(depth_vals - camera[2][valid_ind_mask]).le(self.accuracy)
        if not depth_mask.any():
            return None

        ind_update = ind_points[valid_ind_mask]
        ind_update = ind_update[depth_mask]
        indices_3d = ind_update.new(num_points + 1).fill_(0)
        indices_2d = ind_update.new(num_points + 1).fill_(0)
        indices_3d[0] = ind_update.shape[0]
        indices_2d[0] = ind_update.shape[0]
        indices_3d[1:1 + indices_3d[0]] = ind_update
        indices_2d[1:1 + indices_2d[0]] = torch.index_select(valid_image_ind, 0, torch.nonzero(depth_mask)[:, 0])

        return indices_3d, indices_2d

    @torch.no_grad()
    def project(self, label, lin_indices_3d, lin_indices_2d, num_points):
        
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]

        output = label.new(num_label_ft, num_points).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        
        return output


class Projection(Function):

    @staticmethod
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, num_points):
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]

        output = label.new(num_label_ft, num_points).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_label = grad_output.clone()
        num_ft = grad_output.shape[0]
        grad_label.resize_(num_ft, 32, 41)
        lin_indices_3d, lin_indices_2d = ctx.saved_variables
        num_ind = lin_indices_3d.data[0]
        vals = torch.index_select(grad_output.data.contiguous().view(num_ft, -1), 1, lin_indices_3d.data[1:1+num_ind])
        grad_label.data.view(num_ft, -1)[:, lin_indices_2d.data[1:1+num_ind]] = vals
        
        return grad_label, None, None, None
