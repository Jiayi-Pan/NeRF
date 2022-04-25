import torch
from torch import Tensor
import torch.nn.functional as F


def compute_rays(img_dim: tuple, int_mat: Tensor, mat_c2w: Tensor) -> tuple:
    """
    Compute rays for each pixel of an image

    Args:
        img_dim (tuple[int, int]): image dimension (h,w)
        int_mat (Tensor): camera intrinsic matrix
        mat_c2w (Tensor): transformation matrix from camera to world frame

    Returns:
        tuple of (
            rays_o: origin of the output rays HxWx3
            rays_d: direction vector of the output rays HxWx3
        )
    """
    img_h, img_w = img_dim
    i, j = torch.meshgrid(torch.linspace(0, img_w-1, img_w),
                          torch.linspace(0, img_h-1, img_h), indexing="xy")
    # change device
    i = i.to(int_mat)
    j = j.to(int_mat)
    # pixel -> cam frame
    fx, fy = int_mat[0, 0], int_mat[1, 1]
    cx, cy = int_mat[0, 2], int_mat[1, 2]
    # print(fx, fy, cx, cy)
    rays_d: Tensor = torch.stack(
        [(i-cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], dim=-1)
    # cam frame -> world
    rays_d = torch.sum(rays_d[..., None, :] * mat_c2w[:3, :3], dim=-1)
    # print(rays_d.shape)
    # rays_d = F.normalize(rays_d, dim=2)

    # origin of the rays
    rays_o = mat_c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def queries_from_rays(
    rays_ori_xyz: torch.Tensor,
    rays_dir_xyz: torch.Tensor,
    sample_thresh: tuple,
    num_samples: int
) -> tuple:
    """Compute the input queries to the NeRF model for the given rays in an image of shape (H,W)

    Args:
        rays_ori_xyz (torch.Tensor): starting location for each ray, shape (W, H, 3)
        rays_dir_xyz (torch.Tensor): normalized direction for each ray, shape (W, H, 3)
        sample_thresh (tuple[float, float]): (nearest, farthest) threshold
        num_samples (int): number of samples to generate

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
                query points along each ray, shape (W, H, num_samples, 3)
                sampled depth value for each ray (W,H,num_samples)
    """
    this_device = rays_dir_xyz.device
    t_n, t_f = sample_thresh

    # # generate depths
    # uniform_sample_depths = torch.linspace(
    #     t_n, t_f, num_samples, device=this_device)
    # noise_shape = list(rays_ori_xyz.shape[:-1]) + [num_samples]
    # noisy_sample_depth = uniform_sample_depths + \
    #     torch.rand(noise_shape, device=this_device) * \
    #     (t_f-t_n)/(num_samples)

    # # update query input point positions
    # query_points_xyz = rays_ori_xyz.unsqueeze(
    #     dim=2) + rays_dir_xyz.unsqueeze(dim=2) * noisy_sample_depth.unsqueeze(dim=-1)
    # return query_points_xyz, noisy_sample_depth

    depth_values = torch.linspace(t_n,t_f, num_samples).to(rays_ori_xyz)
     # ray_origins: (width, height, 3)
     # noise_shape = (width, height, num_samples)
    noise_shape = list(rays_ori_xyz.shape[:-1]) + [num_samples]
    # depth_values: (num_samples)
    depth_values = depth_values \
        + torch.rand(noise_shape).to(rays_ori_xyz) * (t_f
            - t_n) / num_samples
    # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
    # query_points:  (width, height, num_samples, 3)
    query_points = rays_ori_xyz[..., None, :] + rays_dir_xyz[..., None, :] * depth_values[..., :, None]
    # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
    return query_points, depth_values


def render_from_nerf(
    nerf_output: torch.Tensor,
    sampled_depths: torch.Tensor
) -> tuple:
    """ Compute the rendering result from the output of NeRF

    Args:
        nerf_output (torch.Tensor): nerf output in rgba, shape (W, H, num_samples, 4)
        sampled_depths (torch.Tensor): sampled depth value for each ray (num_samples)

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                rgb_img, shape (W, H, 3)
                depth_img, shape (W, H)
    """
    # normalize the nerf output
    normed_raw_alpha_maps = F.relu(nerf_output[..., 3])
    # TODO: why this?
    # rgb_maps = torch.sigmoid(nerf_output[..., :3])
    rgb_maps = nerf_output[..., :3]

    # span for each sampled points by diff
    depth_spans = sampled_depths[..., 1:] - sampled_depths[..., :-1]
    depth_spans = torch.cat(
        [depth_spans, torch.Tensor([1e10]).to(depth_spans).expand_as(sampled_depths[...,:1])], dim=-1)
    # compute the weights for each point in a pixel by alpha and depth_span
    alpha_maps = 1. - torch.exp(- normed_raw_alpha_maps * depth_spans)

    # get exp(-\sum \delta * \sigma)
    transmit_acc = torch.cumprod(1-alpha_maps + 1e-10, dim=-1)
    transmit_acc = transmit_acc.roll(shifts=1, dims=-1)
    transmit_acc[..., 0] = 1
    weights = alpha_maps * transmit_acc

    # results
    rgb_img = (rgb_maps * weights.unsqueeze(-1)).sum(dim=-2)
    depth_img = (sampled_depths * weights).sum(dim=-1)
    return rgb_img, depth_img
