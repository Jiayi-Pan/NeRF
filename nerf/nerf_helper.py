import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from nerf.graphics import *


def nerf_iter_once(model: nn.Module,
                  img_dim: tuple[int, int],
                  int_mat: Tensor,
                  mat_c2w: Tensor,
                  sample_thresh: tuple[float, float],
                  num_samples: int = 16,
                  batch_size=10)->tuple(Tensor, Tensor):
    """
    iterate one time

    Args:
        model (nn.Module): nerf model to use
        img_dim (tuple[int, int]): dimension of image (H, W)
        int_mat (Tensor): camera intrinsic matrix
        mat_c2w (Tensor): transformation matrix from camera to world
        sample_thresh (tuple[float, float]): range of sampling (near, far)
        num_samples (int, optional): number of samples points per ray. Defaults to 16.
        batch_size (int, optional): number of pixels to train per batch. Defaults to 10.

    Returns:
        tuple of (
            rgb image: Tensor of shape HxWx3
            depth image: Tensor of shape HxW
        )
    """
    # compute rays on all pixels of the image
    rays_o, rays_d = compute_rays(img_dim, int_mat, mat_c2w)

    # Sample points on each ray
    samples, depth_values = queries_from_rays(rays_o, rays_d, sample_thresh,
                                              num_samples)

    # flatten samples
    samples = samples.reshape(-1, 3)

    # encode points
    # TODO: set encoding dimension as parameter
    encoded_samples = PosEncode(samples, 10)
    encoded_dirs = PosEncode(rays_d, 4)

    # train each pixel
    nerf_out_list: list = []
    n = len(encoded_dirs)
    for i in range(0, n, batch_size):
        if (i + batch_size > n):
            samples_crop = encoded_samples[num_samples * i:]
            dirs_crop = encoded_dirs[i:].expand(samples_crop.shape)
        else:
            samples_crop = encoded_samples[num_samples * i: num_samples * (i+batch_size)]
            dirs_crop = encoded_dirs[i:i+batch_size].expand(samples_crop.shape)

        nerf_out_list.append(
                model(samples_crop, dirs_crop))
    
    nerf_out = torch.cat(nerf_out_list, dim=0)

    nerf_out = nerf_out.reshape(img_dim[0], img_dim[1], num_samples, 4)
    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_predicted, depth_img = render_from_nerf(nerf_out, rays_o, depth_values)

    return rgb_predicted, depth_img


def PosEncode(x, L):
    '''
    Mapping inputs to a higher dimensional space using
    high frequency functions enables better fitting of data
    x -> (sin(2^0 x), cos(2^0 x), ... , sin(2^(L-1) x), cos(2^(L-1) x))

    args:
        x: (N, 3)
        L: half the expanded dimension
            In the original paper, L=10 for position, L=4 for direction
    
    out:
        enc_x: (N, 3*2*L)
    '''
    N = x.shape[0]
    enc_x = torch.zeros((N, 3, 2 * L), dtype=x.dtype, device=x.device)

    # Normalize x
    sum_x = torch.sqrt(torch.sum(x**2, dim=1)).reshape(N, 1)
    x /= sum_x

    # encode
    freq_band = 2**torch.floor(torch.arange(0, L, 0.5, device=x.device))
    freq_t_x = freq_band.reshape(1, -1) * x.reshape(
        N, -1, 1)  # freq_band*x (N, 3, 2L)

    enc_x[:, :, 0::2] = torch.sin(freq_t_x[:, :, 0::2])
    enc_x[:, :, 1::2] = torch.cos(freq_t_x[:, :, 1::2])
    enc_x = torch.flatten(enc_x, start_dim=1)
    return enc_x