import numpy as np
import torch
import json
import os.path
import cv2

def load_blender(data_path: str, scale_factor: int = 0, data_type="train", device="cpu") -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    load data from blender datasets

    Args:
        data_path (str): path to dataset
        scale_factor (int, optional): scale image pixels using H << scale_factor. Defaults to 0.
        data_type (str, optional): type of data to load, str in ("train", "val", "test"). Defaults to "train".
        device (str, optional): device to be used. Defaults to "cpu".

    Returns:
        tuple of (
                 imgs: tensor of shape NxWxHxC
                 poses: tensor of shape Nx4x4
                 int_mat: intrinsic matrix of shape 3x3
                 )
    """
    assert (data_type in ("train", "val", "test"))

    json_path: str = os.path.join(data_path, "transforms_{}.json".format(data_type))
    with open(json_path) as f:
        meta: dict = json.load(f)

    # get n, h, w
    frames: list[dict] = meta["frames"]
    n: int = len(frames)
    test_image = cv2.imread(os.path.join(data_path, "{}.png".format(frames[0]["file_path"])))
    h, w = test_image.shape[:2]
    img_list = []
    poses: torch.Tensor = torch.zeros((n, 4, 4), device=device, dtype=torch.float64)
    for i, frame in enumerate(frames):
        img_path: str = os.path.join(data_path, "{}.png".format(frame['file_path']))
        # convert to rgb order
        img_list.append(cv2.cvtColor(
            cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA))
        poses[i] = torch.Tensor(frame["transform_matrix"])

    # resize image
    if scale_factor != 0:
        scale = (1 << scale_factor)
        h //= scale
        w //= scale
        for i in range(len(img_list)):
            img_list[i] = cv2.resize(img_list[i], (h, w), interpolation=cv2.INTER_AREA)

    # turn image into tensor
    imgs_np = np.stack(img_list, axis=0)
    imgs: torch.Tensor = torch.from_numpy(imgs_np)
    imgs = imgs.to(device=device, dtype=torch.float64)
    imgs/=255

    # calculate focal length
    aov = meta["camera_angle_x"]
    focal_length: float = 0.5 * w / np.tan(0.5 * aov)

    # intrinsic matrix
    int_mat: torch.Tensor = torch.Tensor(
        [[focal_length, 0, w / 2],
         [0, focal_length, h / 2],
         [0, 0, 1]])
    int_mat = int_mat.to(device=device, dtype=torch.float64)

    return imgs, poses, int_mat

