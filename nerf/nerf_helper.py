import torch
import torch.utils.data
from torch import nn, scalar_tensor
from torch.nn import functional as F
from torch import Tensor
from nerf.graphics import *


def nerf_iter_once(model: nn.Module,
                  img_dim: tuple,
                  int_mat: Tensor,
                  mat_c2w: Tensor,
                  sample_thresh: tuple,
                  L_pos=10,
                  L_dir=4,
                  num_samples: int = 16,
                  batch_size=10)->tuple:
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
    encoded_samples = PosEncode(samples, L_pos, True)
    encoded_dirs = PosEncode(rays_d.reshape(-1,3), L_dir, True)
    # print(encoded_dirs.shape)
    # print(encoded_samples.shape)
    # train each pixel
    nerf_out_list: list = []
    n = len(encoded_dirs)
    for i in range(0, n, batch_size):
        if (i + batch_size > n):
            samples_crop = encoded_samples[num_samples * i:]
            dirs_crop = encoded_dirs[i:].repeat(num_samples,1)
        else:
            samples_crop = encoded_samples[num_samples * i: num_samples * (i+batch_size)]
            dirs_crop = encoded_dirs[i:i+batch_size].repeat(num_samples,1)
        
        # change device
        samples_crop = samples_crop.to(mat_c2w)
        dirs_crop = dirs_crop.to(mat_c2w)

        nerf_out_list.append(
                model(samples_crop, dirs_crop)
        )
    
    nerf_out = torch.cat(nerf_out_list, dim=0)

    nerf_out = nerf_out.reshape(img_dim[0], img_dim[1], num_samples, 4)
    # Perform differentiable volume rendering to re-synthesize the RGB image.
    depth_values = depth_values.to(nerf_out)
    rgb_predicted, depth_img = render_from_nerf(nerf_out, depth_values)
    # print(rgb_predicted[50,50])

    return rgb_predicted, depth_img


def PosEncode(x, L, include_itself=False):
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

    # Normalize x
    # sum_x = torch.sqrt(torch.sum(x**2, dim=1)).reshape(N, 1)
    # xx = x / sum_x
    xx = x

    # encode
    if include_itself:
        enc_x = torch.zeros((N, 3, 2 * L + 1), dtype=x.dtype, device=x.device)
        freq_band = 2 ** torch.floor(torch.arange(0, L, 0.5, device=x.device))
        freq_band = torch.cat((torch.tensor([0], device=x.device), freq_band))
        
        freq_t_x = freq_band.reshape(1, -1) * xx.reshape(N, -1, 1)  # freq_band*x (N, 3, 2L)

        enc_x[:, :, 0] = x
        enc_x[:, :, 1::2] = torch.sin(freq_t_x[:, :, 1::2])
        enc_x[:, :, 2::2] = torch.cos(freq_t_x[:, :, 2::2])
        # enc_x = enc_x.reshape(N, 6*L+3)


    else:
        enc_x = torch.zeros((N, 3, 2 * L), dtype=x.dtype, device=x.device)
        freq_band = 2**torch.floor(torch.arange(0, L, 0.5, device=x.device))
        freq_t_x = freq_band.reshape(1, -1) * xx.reshape(N, -1, 1)  # freq_band*x (N, 3, 2L)

        enc_x[:, :, 0::2] = torch.sin(freq_t_x[:, :, 0::2])
        enc_x[:, :, 1::2] = torch.cos(freq_t_x[:, :, 1::2])
        # enc_x = enc_x.reshape(N, 6*L)
    enc_x = torch.flatten(enc_x, start_dim=1)
    
    return enc_x


def compare(**kwargs):
    from nerf.data import load_blender
    from matplotlib import pyplot as plt
    imgs = kwargs["imgs"]
    poses = kwargs["poses"]
    config = kwargs["config"]
    scale_factor = kwargs["scale_factor"]
    DEVICE = kwargs["DEVICE"]
    model = kwargs["model"]
    img_h, img_w = imgs[1:3]
    sample_t = kwargs["sample_t"]
    L_pos = kwargs["L_pos"]
    L_dir = kwargs["L_dir"]
    num_samples = kwargs["num_samples"]
    batch_size = kwargs["batch_size"]

    idx = [78, 84, 90, 158] #hotdog, train, train, test, test
    # idx = [6, 52, 38, 87]  # lego

    imgs_test, poses_test, int_mat_test = load_blender(config.datadir, data_type="test",scale_factor=scale_factor, device="cpu")

    c2w0 = poses[idx[0]].clone().to(DEVICE)
    c2w1 = poses[idx[1]].clone().to(DEVICE)
    c2w2 = poses_test[idx[2]].clone().to(DEVICE)
    c2w3 = poses_test[idx[3]].clone().to(DEVICE)

    pred0, _ = nerf_iter_once(
                    model,
                    (img_h, img_w),
                    int_mat_test.to(DEVICE),
                    c2w0,
                    sample_t,
                    L_pos,
                    L_dir,
                    num_samples=num_samples,
                    batch_size=batch_size
            )
    pred1, _ = nerf_iter_once(
                    model,
                    (img_h, img_w),
                    int_mat_test.to(DEVICE),
                    c2w1,
                    sample_t,
                    L_pos,
                    L_dir,
                    num_samples=num_samples,
                    batch_size=batch_size
            )
    pred2, _ = nerf_iter_once(
                    model,
                    (img_h, img_w),
                    int_mat_test.to(DEVICE),
                    c2w2,
                    sample_t,
                    L_pos,
                    L_dir,
                    num_samples=num_samples,
                    batch_size=batch_size
            )
    pred3, _ = nerf_iter_once(
                    model,
                    (img_h, img_w),
                    int_mat_test.to(DEVICE),
                    c2w3,
                    sample_t,
                    L_pos,
                    L_dir,
                    num_samples=num_samples,
                    batch_size=batch_size
            )

    plt.figure(figsize=(8, 16))
    plt.subplot(421)
    img_np = pred0.detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(f"Pred img: Train {idx[0]}")
    plt.subplot(422)
    plt.imshow(imgs[idx[0]])
    plt.title(f"Ground Truth img: Train {idx[0]}")
    plt.subplot(423)
    img_np = pred1.detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(f"Pred img: Train {idx[1]}")
    plt.subplot(424)
    plt.imshow(imgs[idx[1]])
    plt.title(f"Ground Truth img: Train {idx[1]}")
    
    plt.subplot(425)
    img_np = pred2.detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(f"Pred img: Test {idx[2]}")
    plt.subplot(426)
    plt.imshow(imgs_test[idx[2]])
    plt.title(f"Ground Truth img: Test {idx[2]}")
    plt.subplot(427)
    img_np = pred3.detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(f"Pred img: Test {idx[3]}")
    plt.subplot(428)
    plt.imshow(imgs_test[idx[3]])
    plt.title(f"Ground Truth img: Test {idx[3]}")
    
    plt.savefig("hotdog_cmp.png")