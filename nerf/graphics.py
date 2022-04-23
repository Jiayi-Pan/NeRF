import torch
from torch import Tensor

def compute_rays(img_h:int, img_w:int, int_mat:Tensor, mat_c2w:Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rays for each pixel of an image

    Args:
        img_h (int): image height
        img_w (int): image width
        int_mat (Tensor): camera intrinsic matrix
        mat_c2w (Tensor): transformation matrix from camera to world frame

    Returns:
        tuple of (
            rays_o: origin of the output rays Nx3
            rays_d: direction vector of the output rays Nx3
        )
    """
    i, j = torch.meshgrid(torch.linspace(0, img_w-1, img_w), torch.linspace(0, img_h-1 ,img_h))
    # transpose and change device
    i = i.T.to(int_mat)
    j = j.T.to(int_mat)
    # pixel -> cam frame
    fx, fy = int_mat[0,0], int_mat[1,1]
    cx, cy = int_mat[0,2], int_mat[1,2]
    rays_d:Tensor = torch.stack([(i-cx)/fx, -(j - cy)/fy, -torch.ones_like(i)]).permute(1,2,0)
    # cam frame -> world
    rays_d = rays_d[..., None, :].matmul(mat_c2w[:3,:3])[:,:,0,:]
    
    # origin of the rays
    rays_o = mat_c2w[:3,-1].expand(rays_d.shape)
    
    return rays_o, rays_d