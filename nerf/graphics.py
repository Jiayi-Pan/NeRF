import torch
from torch import Tensor
import torch.nn.functional as F

def compute_rays(img_dim: tuple[int, int], int_mat:Tensor, mat_c2w:Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    i, j = torch.meshgrid(torch.linspace(0, img_w-1, img_w), torch.linspace(0, img_h-1 ,img_h), indexing="xy")
    # change device
    i = i.to(int_mat)
    j = j.to(int_mat)
    # pixel -> cam frame
    fx, fy = int_mat[0,0], int_mat[1,1]
    cx, cy = int_mat[0,2], int_mat[1,2]
    rays_d:Tensor = torch.stack([(i-cx)/fx, -(j - cy)/fy, -torch.ones_like(i)]).permute(1,2,0)
    # cam frame -> world
    rays_d = rays_d[..., None, :].matmul(mat_c2w[:3,:3])[:,:,0,:]
    rays_d = F.normalize(rays_d, dim=2)
    
    # origin of the rays
    rays_o = mat_c2w[:3,-1].expand(rays_d.shape)
    
    return rays_o, rays_d


def queries_from_rays(
    rays_ori_xyz: torch.Tensor,
    rays_dir_xyz: torch.Tensor,
    sample_thresh: tuple[float, float],
    num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the input queries to the NeRF model for the given rays in an image of shape (H,W)
    
    Args:
    	rays_ori_xyz (torch.Tensor): starting location for each ray, shape (W, H, 3)
    	rays_dir_xyz (torch.Tensor): normalized direction for each ray, shape (W, H, 3)
        sample_thresh (tuple[float, float]): (nearest, farthest) threshold
    	num_samples (int): number of samples to generate
    
    Returns:
    	tuple[torch.Tensor, torch.Tensor]:
    		query points along each ray, shape (W, H, num_samples, 3)
    		sampled depth value for each ray (num_samples)
    """
    this_device = rays_dir_xyz.device
    t_n, t_f = sample_thresh
    
    # generate depths
    uniform_sample_depths = torch.linspace(t_n, t_f, num_samples, device=this_device)
    depth_noise = uniform_sample_depths + torch.rand(uniform_sample_depths.shape, device=this_device)*(t_f-t_n)/(2*num_samples)
    noisy_sample_depth = uniform_sample_depths + depth_noise
    
    # update query input point positions
    query_points_xyz =  rays_ori_xyz.unsqueeze(dim=2) + rays_dir_xyz.unsqueeze(dim=2)* noisy_sample_depth.unsqueeze(dim=-1)
    return query_points_xyz, noisy_sample_depth


def render_from_nerf(
    nerf_output: torch.Tensor,
    rays_ori_xyz: torch.Tensor,
    sampled_depths: torch.Tensor
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Compute the rendering result from the output of NeRF
    
    Args:
    	nerf_output (torch.Tensor): nerf output in rgba, shape (W, H, num_samples, 4)
    	rays_ori_xyz (torch.Tensor): starting location for each ray, shape (W, H, 3)
    	sampled_depths (torch.Tensor): sampled depth value for each ray (num_samples)
    
    Returns:
    	tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
    		rgb_img, shape (W, H, 3)
    		depth_img, shape (W, H)
    """	
    # normalize the nerf output
    normed_raw_alpha_maps = F.relu(nerf_output[..., 3])
    rgb_maps = F.sigmoid(nerf_output[..., :3])
    
    # span for each sampled points by diff
    depth_spans = sampled_depths[1:] - sampled_depths[:-1]
    depth_spans = torch.cat(depth_spans, torch.Tensor(1e10))
    
    # compute the weights for each point in a pixel by alpha and depth_span
    alpha_maps = 1. - torch.exp(- normed_raw_alpha_maps * depth_spans)
    
    # get exp(-\sum \delta * \sigma)
    cum_prods = torch.cumprod(1-alpha_maps + 1e-10, dim=-1)
    cum_prods = cum_prods.roll(shifts=1, dim=-1)
    cum_prods[..., -1] = 0
    weights = alpha_maps * cum_prods
    
    # results
    rgb_img = (rgb_maps * weights.unsqueeze(-1)).sum(dim=-2)
    depth_img = (sampled_depths * weights).sum(dim=-1)
    return rgb_img, depth_img