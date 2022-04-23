import torch
from torch.nn import functional as F

def queries_from_rays(
	rays_ori_xyz: torch.Tensor,
	rays_dir_xyz: torch.Tensor,
	t_n: float,
	t_f: float,
	num_samples: int
	) -> tuple[torch.Tensor, torch.Tensor]:
	"""Compute the input queries to the NeRF model for the given rays in an image of shape (H,W)

	Args:
		rays_ori_xyz (torch.Tensor): starting location for each ray, shape (W, H, 3)
		rays_dir_xyz (torch.Tensor): normalized direction for each ray, shape (W, H, 3)
		t_n (float): nearest threshold
		t_f (float): farthest threhold
		num_samples (int): number of samples to generate

	Returns:
		tuple[torch.Tensor, torch.Tensor]:
			query points along each ray, shape (W, H, num_samples, 3)
			sampled depth value for each ray (num_samples)
	"""
	this_device = rays_dir_xyz.device

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
		nerf_output (torch.Tensor): nerf output in rgba, shape (W, H, 4)
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

	# TODO
	# weights = alpha_maps

	# results
	rgb_img = (rgb_maps * weights.unsqueeze(-1)).sum(dim=-2)
	depth_img = (sampled_depths * weights).sum(dim=-1)
	return rgb_img, depth_img