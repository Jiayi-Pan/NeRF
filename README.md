# A Guided Tour to Nerual Radiance Field

## Quick start
```bash
conda env create -f environment.yml
conda activate nerf
```
## Data

### Blender

+ `camera_angle_x` - angle of view (AOV) of the camera

+ `frames`
  + `file_path` - path of the frame
  + `rotation` - not been used...
  + `transform_matrix` - transform cam_frame -> world_frame

## Usage

1. Download `nerf_synthetic.zip` from [Nerf Data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1?usp=sharing), unzip under `./data/`
2. Follow the instructions in `NeRF.ipynb`