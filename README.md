# Dense Depth Priors for NeRF from Sparse Input Views
This repository contains the implementation of the CVPR 2022 paper: Dense Depth Priors for Neural Radiance Fields from Sparse Input Views.

[Arxiv](https://arxiv.org/abs/2112.03288) | [Video](https://t.co/zjH9JvkuQq) | [Project Page](https://barbararoessle.github.io/dense_depth_priors_nerf/)

![](docs/static/images/pipeline.jpg)

## Step 1: Train Dense Depth Priors
You can skip this step and download the depth completion model trained on ScanNet from [here](https://drive.google.com/drive/folders/1HTyigHPJKZKBWzGFoY8J2bcS-h8_SfX9?usp=sharing). 

### Prepare ScanNet
Extract the ScanNet dataset e.g. using [SenseReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) and place the files `scannetv2_test.txt`, 
`scannetv2_train.txt`, `scannetv2_val.txt` from [ScanNet Benchmark](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark) into the same directory. 

### Precompute Sampling Locations
Run the [COLMAP](https://github.com/colmap/colmap) feature extractor on all RGB images of ScanNet. 
For this, the RGB files need to be isolated from the other scene data, f.ex. create a temporary directory `tmp` and copy each `<scene>/color/<rgb_filename>` to `tmp/<scene>/color/<rgb_filename>`. 
Then run: 
```
colmap feature_extractor  --database_path scannet_sift_database.db --image_path tmp
```
When working with different relative paths or filenames, the database reading in `scannet_dataset.py` needs to be adapted accordingly. 

### Download pretrained ResNet
Download the pretrained ResNet from [here](https://drive.google.com/file/d/17adZHo5dkcU8_M_6OvYzGUTDguF6k-Qu/view) . 

### Train
```
python3 run_depth_completion.py train --dataset_dir <path to ScanNet> --db_path <path to database> --pretrained_resnet_path <path to pretrained resnet> --ckpt_dir <path to write checkpoints>
```
Checkpoints are written into a subdirectory of the provided checkpoint directory. The subdirectory is named by the training start time in the format `jjjjmmdd_hhmmss`, which also serves as experiment name in the following. 

### Test
```
python3 run_depth_completion.py test --expname <experiment name> --dataset_dir <path to ScanNet> --db_path <path to database> --ckpt_dir <path to write checkpoints>
```

## Step 2: Optimizing NeRF with Dense Depth Priors
### Prepare scenes
You can skip the scene preparation and directly download the [scenes](https://drive.google.com/drive/folders/1vJ5sZaYljmaxMc1vltm6u4GUH11oqfYU?usp=sharing). 
To prepare a scene and render sparse depth maps from COLMAP sparse reconstructions, run: 
```
cd preprocessing
mkdir build
cd build
cmake ..
make -j
./extract_scannet_scene <path to scene> <path to ScanNet>
```
The scene directory must contain the following:
- `train.csv`: list of training views from the ScanNet scene
- `test.csv`: list of test views from the ScanNet scene
- `config.json`: parameters for the scene:
  - `name`: name of the scene
  - `max_depth`: maximal depth value in the scene, larger values are invalidated
  - `dist2m`: scaling factor that scales the sparse reconstruction to meters
  - `rgb_only`: write RGB only, f.ex. to get input for COLMAP
- `colmap`: directory containing 2 sparse reconstruction:
  - `sparse`: reconstruction run on train and test images together to determine the camera poses
  - `sparse_train`, reconstruction run on train images alone to determine the sparse depth maps.  

Please check the provided scenes as an example. 
The option `rgb_only` is used to preprocess the RGB images before running COLMAP. This cuts dark image borders from calibration, which harm the NeRF optimization. It is essential to crop them before running COLMAP to ensure that the determined intrinsics match the cropped RGB images. 

### Optimize
```
python3 run_nerf.py train --scene_id <scene, e.g. scene0710_00> --data_dir <directory containing the scenes> --depth_prior_network_path <path to depth prior checkpoint> --ckpt_dir <path to write checkpoints>
```
Checkpoints are written into a subdirectory of the provided checkpoint directory. The subdirectory is named by the training start time in the format `jjjjmmdd_hhmmss`, which also serves as experiment name in the following. 

### Test
```
python3 run_nerf.py test --expname <experiment name> --data_dir <directory containing the scenes> --ckpt_dir <path to write checkpoints>
```
The test results are stored in the experiment directory. 
Running `python3 run_nerf.py test_opt ...` performs test time optimization of the latent codes before computing the test metrics. 

### Render Video
```
python3 run_nerf.py video  --expname <experiment name> --data_dir <directory containing the scenes> --ckpt_dir <path to write checkpoints>
```
The video is stored in the experiment directory. 

---

### Citation
If you find this repository useful, please cite: 
```
@inproceedings{roessle2022depthpriorsnerf,
    title={Dense Depth Priors for Neural Radiance Fields from Sparse Input Views}, 
    author={Barbara Roessle and Jonathan T. Barron and Ben Mildenhall and Pratul P. Srinivasan and Matthias Nie{\ss}ner},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022}
```

### Acknowledgements
We thank [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [CSPN](https://github.com/XinJCheng/CSPN), from which this repository borrows code. 
