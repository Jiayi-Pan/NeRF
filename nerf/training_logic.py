import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from nerf.nerf_helper import *


def train_NeRF(model, optimizer, imgs_train, imgs_val, poses_train, poses_val, int_mat, sample_t, L_pos, L_dir, num_samples, ckpt_path, batch_size, psnrs, val_its, start_iter_num=0, end_iter_num=10000, val_gap=200, DEVICE='cuda'):
    psnrs = []
    val_its = []
    for iter_num in tqdm(range(start_iter_num, end_iter_num)):
        if iter_num % val_gap == 0:
            val_one_step(iter_num, model, optimizer, imgs_val, poses_val, int_mat,
                         sample_t, L_pos, L_dir, psnrs, val_its, num_samples, ckpt_path, batch_size, DEVICE)
        else:
            train_one_step(model, optimizer, imgs_train, poses_train, int_mat,
                           sample_t, L_pos, L_dir, num_samples, batch_size, DEVICE)


def train_one_step(model, optimizer, imgs_train, poses_train, int_mat, sample_t, L_pos, L_dir, num_samples, batch_size, DEVICE):
    gt_img_idx = np.random.randint(100)
    gt_img = imgs_train[gt_img_idx].clone().to(DEVICE)
    gt_c2w = poses_train[gt_img_idx].clone().to(DEVICE)

    img_h, img_w, _ = gt_img.shape
    optimizer.zero_grad()
    pred_rgb, _ = nerf_iter_once(
        model,
        (img_h, img_w),
        int_mat.to(DEVICE),
        gt_c2w,
        sample_t,
        L_pos,
        L_dir,
        num_samples=num_samples,
        batch_size=batch_size
    )
    loss = F.mse_loss(pred_rgb, gt_img[..., :3])
    loss.backward()
    optimizer.step()


def val_one_step(iter_num, model, optimizer, imgs_val, poses_val, int_mat, sample_t, L_pos, L_dir, psnrs, val_its, num_samples, ckpt_path, batch_size, DEVICE):
    val_idx = np.random.randint(imgs_val.shape[0])
    val_img = imgs_val[val_idx].clone().to(DEVICE)
    val_c2w = poses_val[val_idx].clone().to(DEVICE)

    img_h, img_w, _ = val_img.shape
    pred_rgb, _ = nerf_iter_once(
        model,
        (img_h, img_w),
        int_mat.to(DEVICE),
        val_c2w,
        sample_t,
        L_pos,
        L_dir,
        num_samples=num_samples,
        batch_size=batch_size
    )

    loss = torch.nn.functional.mse_loss(pred_rgb, val_img[..., :3])
    print("Iteration ", iter_num)
    print("Val loss: ", loss)

    psnr = -10. * torch.log10(loss)
    psnrs.append(psnr.item())
    val_its.append(iter_num)

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    img_np = pred_rgb.detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(f"Iteration {iter_num}")
    plt.subplot(122)
    plt.plot(val_its, psnrs)
    plt.title("PSNR")
    plt.show()

    torch.save({
        'epoch': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'psnrs': psnrs,
        'its': val_its}, ckpt_path)
