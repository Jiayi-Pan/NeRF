expname = "blender_lego"
basedir = "./logs"
datadir = "./data/nerf_synthetic/lego"
dataset_type = "blender"

no_batching = True

use_viewdirs = True
white_bkgd = True
lrate_decay = 500

N_samples = 64
N_importance = 128
N_rand = 1024

precrop_iters = 500
precrop_frac = 0.5

half_res = True
