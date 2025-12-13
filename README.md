# 3D Object Detection Inference Pipeline

This repository contains scripts for running 3D object detection inference on KITTI and nuScenes datasets using MMDetection3D models.

View the demo images and videos [here](https://drive.google.com/drive/folders/1LiejW3UdqJp7DUHIKjqOU4D3ugv97MaQ?usp=sharing).

## Setup

### 1. Environment Setup

```bash
# Activate the conda environment
conda activate openmmlab2

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Checkpoint Files

Download model checkpoints to `mmdetection3d/checkpoints/`:
- PointPillars KITTI: `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth`
- Second KITTI: `hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth`
- PointPillars nuScenes: `hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth`
- CenterPoint nuScenes: `centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth`

## Usage

### Workflow: Inference → Generate Images → Create Videos

The complete workflow consists of 3 steps:

1. **Run Inference**: Generates PLY files (point clouds + bounding boxes) and label JSON files
2. **Generate Visualization Images**: Converts PLY files to PNG images with labels
3. **Create Video**: Combines images into a video file

**Note:** The `--max-samples` flag limits the number of samples processed. Use `-1` or omit it to process the full dataset:
- **KITTI validation set**: ~3,769 samples
- **nuScenes validation set**: ~6,019 samples (or 914 in some splits)

### General Command Template

```bash
cd /home/student/cmpe276_3d_detection/students/s017530084

# Step 1: Run inference
python simple_infer_main.py \
  --config <config_file> \
  --checkpoint <checkpoint_file> \
  --dataroot <data_root> \
  --ann-file <annotation_file> \
  --out-dir <output_directory> \
  --dataset <kitti|nuscenes> \
  --data-source cfg \
  --max-samples 50

# Step 2: Generate visualization images from PLY files
python open3d_view_saved_ply.py \
  --dir <output_directory>/<timestamp> \
  --all \
  --headless

# Step 3: Create video from visualization images
python create_video_from_images.py \
  --dir <output_directory>/<timestamp> \
  --num-frames 50 \
  --fps 2.0
```

### Model Configurations

#### KITTI Models

| Model | Config | Checkpoint |
|-------|--------|------------|
| PointPillars | `../../mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py` | `../../mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth` |
| Second | `../../mmdetection3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py` | `../../mmdetection3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth` |

**KITTI Data Paths:**
- `--dataroot`: `/home/student/cmpe276_3d_detection/data/kitti`
- `--ann-file`: `/home/student/cmpe276_3d_detection/data/kitti/kitti_infos_val.pkl`

#### nuScenes Models

| Model | Config | Checkpoint |
|-------|--------|------------|
| PointPillars | `pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d-no-sweeps.py` | `../../mmdetection3d/checkpoints/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth` |
| CenterPoint | `centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d-no-sweeps.py` | `../../mmdetection3d/checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth` |

**nuScenes Data Paths:**
- `--dataroot`: `/home/student/cmpe276_3d_detection/data/nuscenes`
- `--ann-file`: `/home/student/cmpe276_3d_detection/data/nuscenes/nuscenes_infos_val.pkl`

### Example: KITTI PointPillars

```bash
cd /home/student/cmpe276_3d_detection/students/s017530084

# Step 1: Run inference
python simple_infer_main.py \
  --config ../../mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
  --checkpoint ../../mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
  --dataroot /home/student/cmpe276_3d_detection/data/kitti \
  --ann-file /home/student/cmpe276_3d_detection/data/kitti/kitti_infos_val.pkl \
  --out-dir ./results_kitti_pointpillars \
  --dataset kitti \
  --data-source cfg \
  --max-samples 50

# Step 2: Generate visualization images (replace <timestamp> with actual directory)
python open3d_view_saved_ply.py \
  --dir results_kitti_pointpillars/<timestamp> \
  --all \
  --headless

# Step 3: Create video
python create_video_from_images.py \
  --dir results_kitti_pointpillars/<timestamp> \
  --num-frames 50 \
  --fps 2.0
```

### Benchmark Evaluation

```bash
# PointPillars
python simple_infer_main.py \
  --config <config_file> \
  --checkpoint <checkpoint_file> \
  --dataroot <data_root> \
  --ann-file <annotation_file> \
  --out-dir <output_directory> \
  --dataset <kitti|nuscenes> \
  --data-source cfg \
  --eval \
  --eval-backend manual
```

## Key Features

### Visualization Features
- **Height-colored points**: Points colored by Z-coordinate (height)
- **Reduced point weight**: Smaller, more transparent points for better visibility
- **Labeled bounding boxes**: Class names and confidence scores displayed above boxes
- **Thick bounding box lines**: 2.5px line width for better visibility

### Benchmark Features
- **Latency tracking**: Per-sample and aggregate inference time
- **Memory usage**: Peak GPU memory consumption
- **Per-class detections**: Count of detections by class
- **Evaluation metrics**: NDS and mAP for nuScenes, per-class AP for KITTI

## Configuration Files

### Custom Configs (No Sweeps)

For nuScenes without sweep files, use these custom configs:
- `pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d-no-sweeps.py`
- `centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d-no-sweeps.py`

These configs remove the `LoadPointsFromMultiSweeps` transform and adjust `use_dim` to match model expectations.


## Performance Notes

- **Latency**: Typically 80-120ms per sample on Tesla P100
- **Memory**: 3.5-8.5GB GPU memory depending on model
- **Throughput**: ~8-12 FPS

## References

- https://github.com/open-mmlab/mmdetection3d
- https://github.com/lkk688/DeepDataMiningLearning/tree/main

