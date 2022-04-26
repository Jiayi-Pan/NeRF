import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from nerf.nerf_helper import *


def train_NeRF(model, optimizer, imgs_train, imgs_val, poses_train, poses_val, int_mat, sample_t, L_pos, L_dir, num_samples, ckpt_path, batch_size, psnrs, val_its, losses, start_iter_num=0, end_iter_num=10000, val_gap=200, DEVICE='cuda'):
    for iter_num in tqdm(range(start_iter_num, end_iter_num)):
        if iter_num % val_gap == 0:
            with torch.no_grad():
                val_one_step(iter_num, model, optimizer, imgs_val, poses_val, int_mat,
                            sample_t, L_pos, L_dir, psnrs, val_its, losses, num_samples, ckpt_path, batch_size, DEVICE)
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


def val_one_step(iter_num, model, optimizer, imgs_val, poses_val, int_mat, sample_t, L_pos, L_dir, psnrs, val_its, losses, num_samples, ckpt_path, batch_size, DEVICE):
    # if loaded checkpoint, pop last result
    if len(val_its)!=0 and iter_num == val_its[-1]:
        psnrs.pop()
        val_its.pop()
        losses.pop()

    # select validation img
    val_idx = np.random.randint(imgs_val.shape[0])
    val_img = imgs_val[val_idx].clone().to(DEVICE)
    val_c2w = poses_val[val_idx].clone().to(DEVICE)
    
    # predict RGB
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

    # calculate loss
    loss = torch.nn.functional.mse_loss(pred_rgb, val_img[..., :3])
    losses.append(float(loss))
    psnr = -10. * torch.log10(loss)
    psnrs.append(psnr.item())
    val_its.append(iter_num)

    # plot the validation result
    print("Iteration ", iter_num)
    print("Val loss: ", losses[-1])
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    img_np = pred_rgb.detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(f"Iteration {iter_num}")
    plt.subplot(132)
    plt.plot(val_its, psnrs)
    plt.title("PSNR")
    plt.subplot(133)
    plt.plot(val_its, losses)
    plt.title("loss")
    plt.show()

    # save checkpoint
    torch.save({
        'epoch': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'psnrs': psnrs,
        'its': val_its},
        ckpt_path
        )
