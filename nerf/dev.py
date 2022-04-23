import torch

def queries_from_rays(
	rays_ori_xyz: torch.Tensor,
	rays_dir_xyz: torch.Tensor,
	t_n: float,
	t_f: float,
	num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
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
	query_points_xyz =  rays_ori_xyz.unflatten(dim=2) + rays_dir_xyz.unflatten(dim=2)* noisy_sample_depth.unflatten(dim=-1)
	return query_points_xyz, noisy_sample_depth.flatten()

	



	