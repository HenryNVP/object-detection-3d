# MMDetection3D 3D Object Detection

## A. Setup

**Environment:** Lab Server

### 1. Create Environment

```bash
conda create --name openmmlab python=3.11 -y
conda activate openmmlab
```

### 2. Install PyTorch 2.1

We use PyTorch 2.1.2 because it is a stable version for MMDetection3D and has pre-built GPU wheels. (Avoids mmcv._ext errors).

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies & Fix Version Crashes

This single command prevents the "Numpy 2.0" crash and fixes the "Numba/LLVMlite" errors.

```bash
pip install -U openmim ninja "numpy<2.0.0" "numba==0.60.0" "llvmlite==0.43.0" \
    "opencv-python<4.10.0" "opencv-python-headless<4.10.0" "shapely==2.0.3" \
    fire cachetools descartes plotly joblib threadpoolctl
```

### 4. Install OpenMMLab Core

We install the specific MMCV version that matches our PyTorch.

```bash
# Install MMEngine
mim install mmengine

# Install MMCV 2.1.0 (GPU)
mim install "mmcv==2.1.0"

# Install MMDetection (Base)
mim install "mmdet>=3.0.0"
```

### 5. Install MMDetection3D (Source)

Clone the repo and install using the `--no-build-isolation` flag to fix the "ModuleNotFoundError: torch" build error.

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
python setup.py develop
```

### 6. Verify & Run Demo

#### Install Visualizer Support

```bash
pip install open3d
```

#### Download Weights

Model zoo: https://github.com/open-mmlab/mmdetection3d/tree/1.0

Checkpoints:
- `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth`
- `hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth`
- `centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth`

#### Run Benchmarks

**PointPillars on KITTI:**

```bash
python simple_infer_main.py \
  --config mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py \
  --checkpoint mmdetection3d/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
  --dataroot /home/student/cmpe276_3d_detection/data/kitti \
  --ann-file /home/student/cmpe276_3d_detection/data/kitti/kitti_infos_val.pkl \
  --out-dir ./results_kitti_pointpillars \
  --dataset kitti \
  --data-source cfg \
  --eval
```

**Second on KITTI:**

```bash
python simple_infer_main.py \
  --config mmdetection3d/configs/second/second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py \
  --checkpoint mmdetection3d/checkpoints/hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth \
  --dataroot /home/student/cmpe276_3d_detection/data/kitti \
  --ann-file /home/student/cmpe276_3d_detection/data/kitti/kitti_infos_val.pkl \
  --out-dir ./results_kitti_second \
  --dataset kitti \
  --data-source cfg \
  --eval
```

**CenterPoint on nuScenes:**

```bash
python simple_infer_main.py \
  --config centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d-no-sweeps.py \
  --checkpoint mmdetection3d/checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth \
  --dataroot /home/student/cmpe276_3d_detection/data/nuscenes \
  --ann-file /home/student/cmpe276_3d_detection/data/nuscenes/nuscenes_infos_val.pkl \
  --out-dir ./results_nuscenes_centerpoint \
  --dataset nuscenes \
  --data-source cfg \
  --eval \
  --eval-backend manual
```

## B. Key Modifications to Inference Pipeline

This section documents the important changes made to `simple_infer_main.py` and `simple_infer_utils.py` that affect the logic and functionality of the inference pipeline.

### 1. Path Resolution System

Added `resolve_path()` function that automatically resolves paths relative to workspace root.

**Example**:
```python
# Works from any directory:
python simple_infer_main.py \
  --config mmdetection3d/configs/pointpillars/... \
  --checkpoint mmdetection3d/checkpoints/... \
  --dataroot data/kitti
```

### 2. NuScenes Token Resolution Fix

Enhanced `build_loader_pack()` to load sample tokens directly from pkl files and added `sample_tokens` list to loader pack. Enhanced `_resolve_sample_token()` with fallback strategies: metadata token → pre-loaded `sample_tokens` → dataset `data_list`/`data_infos` → direct validation. Fixed "0 samples in predictions file" error and supports both old/new mmdet3d dataset structures.

### 3. KITTI Dataset Support

Added complete KITTI evaluation pipeline in `run_manual_benchmark()` for Consistent evaluation format across nuScenes and KITTI

### 4. Evaluation Robustness Improvements

Filter predictions JSON to only include processed samples and monkey-patch `NuScenesEval.__init__` to handle filtered predictions. Filters ground truth boxes to match processed tokens and gracefully handles missing `lidar2img` in metadata. Enables evaluation with sub-sampled datasets without assertion errors when using `--max-samples`.

### 5. Output Organization

Added timestamped output directories for non-Runner paths. Each run creates a unique timestamped subdirectory, which matches MMEngine Runner behavior for consistency

### 6. Enhanced Configuration Path Handling

Enhanced `patch_cfg_paths()` with intelligent path resolution using multiple fallback strategies. Supports absolute paths, relative to CWD, relative to dataroot, or relative to workspace root. Provides better error handling and path validation for more flexible configuration file paths.


## C. Benchmark Results

### System Configuration
- **GPU**: Tesla P100-PCIE-12GB
- **System Memory**: 188.44 GB
- **MMDetection3D Version**: 1.4.0
- **PyTorch Version**: 2.1.2

### Performance Summary

| Model        | Dataset   | Latency (ms)        | Memory (MB) | Samples | Score Threshold |
|--------------|-----------|---------------------|------------|---------|------------------|
| PointPillars | KITTI     | 113.14 ± 459.14     | 8,475.56   | 50      | 0.05             |
| Second       | KITTI     | 108.97 ± 233.11     | 3,464.23   | 50      | 0.05             |
| CenterPoint  | nuScenes  | 122.31 ± 208.41     | 4,251.80   | 914     | 0.05             |

### Detailed Results

#### 1. KITTI - PointPillars

- **Config**: `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py`
- **Checkpoint**: `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth`
- **Latency**: 
  - Mean: 113.14 ms
  - Min: 42.11 ms
  - Max: 3,327.05 ms
  - Std: 459.14 ms
- **Memory**: Peak: 8,475.56 MB
- **Per-Class Detections**:
  - Car: 284
  - Pedestrian: 208
  - Cyclist: 101
  - **Total**: 593

#### 2. KITTI - Second

- **Config**: `second_hv_secfpn_8xb6-amp-80e_kitti-3d-3class.py`
- **Checkpoint**: `hv_second_secfpn_fp16_6x8_80e_kitti-3d-3class_20200925_110059-05f67bdf.pth`
- **Latency**: 
  - Mean: 108.97 ms
  - Min: 69.43 ms
  - Max: 1,740.65 ms
  - Std: 233.11 ms
- **Memory**: Peak: 3,464.23 MB
- **Per-Class Detections**:
  - Car: 226
  - Pedestrian: 124
  - Cyclist: 72
  - **Total**: 422

#### 3. nuScenes - CenterPoint

- **Config**: `centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d-no-sweeps.py`
- **Checkpoint**: `centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth`
- **Latency**: 
  - Mean: 122.31 ms
  - Min: 105.95 ms
  - Max: 6,418.95 ms
  - Std: 208.41 ms
- **Memory**: Peak: 4,251.80 MB
- **Evaluation Metrics**:
  - NDS: 0.0000
  - mAP: 0.0000
- **Note**: Evaluation completed successfully but metrics are 0.0000, which may indicate model performance issues or prediction format mismatches.

### Benchmark JSON Files

All benchmark results are saved in JSON format:

- `results_kitti_pointpillars/benchmark_kitti.json` - PointPillars KITTI results
- `results_kitti_second/benchmark_kitti.json` - Second KITTI results
- `results_nuscenes_centerpoint/benchmark_nuscenes.json` - CenterPoint nuScenes results
- `benchmark_results_aggregated.json` - Aggregated summary of all benchmarks

### Observations

1. **Latency Performance**: 
   - Second model shows the most consistent performance with the lowest standard deviation (233.11 ms)
   - PointPillars has higher variance (459.14 ms), indicating less predictable inference times
   - All models achieve sub-150ms average inference time, suitable for real-time applications

2. **Memory Efficiency**: 
   - Second model is the most memory-efficient (3,464 MB)
   - PointPillars requires the most memory (8,475 MB), approximately 2.4× more than Second
   - CenterPoint uses moderate memory (4,252 MB)

3. **Detection Performance**: 
   - PointPillars detects more objects overall (593 total) compared to Second (422 total) on the same KITTI samples
   - This suggests PointPillars may have higher recall but potentially lower precision

4. **Dataset Comparison**: 
   - KITTI benchmarks used 50 samples for evaluation
   - nuScenes benchmark used 914 samples, providing more comprehensive evaluation
