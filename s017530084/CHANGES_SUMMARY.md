# Summary of Changes: Modified vs Original Files

## Overview
This document summarizes the key modifications made to `simple_infer_main.py` and `simple_infer_utils.py` in the student directory compared to the original files in the main directory.

---

## 1. `simple_infer_main.py` Changes

### 1.1 Numba CUDA Environment Variables
**Location**: Lines 132-133

**Original**:
```python
# os.environ['NUMBAPRO_LIBDEVICE'] = "..."
# os.environ['NUMBAPRO_NVVM'] = "..."
```

**Modified**:
```python
os.environ['NUMBAPRO_LIBDEVICE'] = "/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/nvvm/libdevice/libdevice.10.bc"
os.environ['NUMBAPRO_NVVM'] = "/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/12.6/nvvm/lib64/libnvvm.so"
```

**Purpose**: Enable Numba CUDA support by uncommenting and activating environment variables for CUDA 12.6 libraries.

### 1.2 Additional Imports
**Location**: Lines 125-126

**Added**:
```python
import time
from datetime import datetime
```

**Purpose**: Support timestamped output directories.

### 1.3 Default `--max-samples` Value
**Location**: Line 228

**Original**: `default=20`
**Modified**: `default=100`

**Purpose**: Increase default sample limit for more comprehensive testing.

### 1.4 Timestamped Output Directories
**Location**: Lines 340-347

**Added**:
```python
# For non-Runner paths, create timestamped subdirectory manually
# (similar to MMEngine Runner behavior)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
timestamped_out_dir = osp.join(args.out_dir, timestamp)
os.makedirs(timestamped_out_dir, exist_ok=True)

# Update args.out_dir to use the timestamped directory
args.out_dir = timestamped_out_dir
print(f"Output directory: {args.out_dir}")
```

**Purpose**: Create timestamped subdirectories for better organization of results, matching MMEngine Runner behavior.

### 1.5 Output Directory Creation Timing
**Location**: Line 322

**Original**: `os.makedirs(args.out_dir, exist_ok=True)` was called before the Runner path check.

**Modified**: Moved to inside the Runner path (line 336) and added timestamped directory creation for non-Runner paths.

**Purpose**: Better organization of output directories.

---

## 2. `simple_infer_utils.py` Changes

### 2.1 New `resolve_path()` Function
**Location**: Lines 1128-1163

**Added**: Complete new function to resolve paths relative to workspace root.

**Purpose**: 
- Allows running scripts from subdirectories while using paths relative to workspace root
- Handles cases like `'mmdetection3d/configs/...'` or `'data/kitti/kitti_infos_val.pkl'`
- Automatically finds workspace root by looking for `mmdetection3d` directory
- Falls back gracefully if path resolution fails

**Key Features**:
- Checks if path is absolute and exists
- Checks if path exists relative to current directory
- Walks up directory tree to find workspace root
- Returns absolute path

### 2.2 Enhanced `patch_cfg_paths()` Function
**Location**: Lines 935-987

**Changes**:
1. **Added `resolve_path()` call for dataroot**:
   ```python
   dataroot = resolve_path(dataroot)
   ```

2. **Enhanced `ann_file` resolution logic**:
   - Checks if `ann_file` is absolute
   - Checks if it exists relative to current directory
   - Tries relative to dataroot
   - Uses `resolve_path()` to try workspace root
   - Better fallback handling

3. **Improved default `ann_file` handling**:
   - Uses `resolve_path()` when joining with dataroot
   - Better path existence checking

**Purpose**: More robust path resolution for configuration files, especially when running from subdirectories.

### 2.3 Enhanced `build_loader_pack()` Function
**Location**: Lines 989-1123

**Key Changes**:

1. **Added `dataset` parameter**:
   ```python
   dataset: str = "nuscenes",
   ```
   **Purpose**: Support both nuScenes and KITTI datasets.

2. **Enhanced sample token loading for nuScenes**:
   - **Original**: Used `dataset_obj.data_infos` directly
   - **Modified**: 
     - Loads tokens directly from pkl file (`data_list` or list format)
     - Falls back to `dataset_obj.data_list` or `dataset_obj.data_infos`
     - Better error handling with try-except
     - Handles both `data_list` (mmdet3d BaseDataset) and `data_infos` (legacy)

3. **Added `sample_tokens` to return dict**:
   ```python
   sample_tokens=sample_tokens,  # <-- NEW
   ```

**Purpose**: 
- Fix "0 samples" issue in nuScenes evaluation
- Ensure correct token resolution for predictions
- Support both old and new mmdet3d dataset structures

### 2.4 Enhanced `load_model_from_cfg()` Function
**Location**: Lines 1166-1203

**Changes**:
- Added `resolve_path()` calls for `config_path` and `checkpoint_path`
- Ensures paths are resolved relative to workspace root before loading

**Purpose**: Support running from subdirectories with relative paths.

### 2.5 Enhanced `run_runner_benchmark()` Function
**Location**: Lines 1206-1236

**Changes**:
- Added `resolve_path()` calls for `config_path` and `checkpoint_path`

**Purpose**: Consistent path resolution across all functions.

### 2.6 Enhanced `run_benchmark_evaluation()` Function
**Location**: Lines 1290-1453

**Key Changes**:

1. **Path Resolution**:
   - Resolves `config_path`, `checkpoint_path`, and `dataroot` using `resolve_path()`
   - Enhanced `ann_file` resolution with multiple fallback strategies

2. **Dataset-Specific Default `ann_file`**:
   ```python
   if args.dataset == 'nuscenes':
       default_ann = 'nuscenes_infos_val.pkl'
   elif args.dataset == 'kitti':
       default_ann = 'kitti_infos_val.pkl'
   ```

**Purpose**: Better support for both nuScenes and KITTI datasets with automatic path resolution.

### 2.7 Enhanced `_resolve_sample_token()` Function
**Location**: Lines 1607-1686

**Key Changes**:

1. **Added `sample_tokens` fallback (Priority 1.5)**:
   ```python
   # 1.5) Try sample_tokens list from pack (if available)
   sample_tokens = pack.get('sample_tokens', None)
   if sample_tokens is not None and len(sample_tokens) > 0:
       try:
           idx = int(loader_token)
           if 0 <= idx < len(sample_tokens):
               tok = sample_tokens[idx]
               if isinstance(tok, str) and len(tok) > 0:
                   return tok
   ```

2. **Changed from `data_infos` to `data_list` or `data_infos`**:
   ```python
   # Try data_list first (mmdet3d BaseDataset), then data_infos (legacy)
   data_list = getattr(ds, 'data_list', None) or getattr(ds, 'data_infos', None)
   ```

**Purpose**: 
- Fix nuScenes token resolution issues
- Support both old (`data_infos`) and new (`data_list`) mmdet3d dataset structures
- Use pre-loaded `sample_tokens` from pkl file for reliability

### 2.8 Enhanced `run_manual_benchmark()` Function
**Location**: Lines 1814-2229

**Key Changes**:

1. **KITTI Support**:
   - Added complete KITTI evaluation path (lines 1859-1985)
   - Per-class detection counting
   - Performance metrics collection
   - Multi-view visualization support
   - Saves `benchmark_kitti.json` with structured results

2. **NuScenes Token Resolution**:
   - Uses `_resolve_sample_token()` for proper token resolution
   - Tracks `processed_tokens` list

3. **NuScenes Evaluation Filtering**:
   - Filters predictions JSON to only processed samples
   - Monkey-patches `NuScenesEval.__init__` to handle filtered predictions
   - Filters GT boxes to match processed tokens
   - Handles "Samples in split doesn't match" assertion error

4. **Enhanced Error Handling**:
   - Better handling of missing `lidar2img` in metadata
   - Graceful fallbacks for token resolution

**Purpose**: 
- Full KITTI benchmark support
- Fix nuScenes evaluation with filtered samples
- Better error handling and robustness

### 2.9 Enhanced `inference_loop()` Function
**Location**: Lines 2589-2719

**Changes**:
1. **Added `max_samples` parameter support**:
   - Properly respects `max_samples` for any data source
   - Limits iteration based on `max_samples` value

2. **Better visualization handling**:
   - Checks for `lidar2img` in metadata before visualization
   - Uses `draw_2d_multiview_from_tensor` for cfg-based loaders
   - Falls back to `draw_2d_multiview` for custom loaders

**Purpose**: Better control over inference loop and visualization.

---

## 3. Summary of Key Improvements

### 3.1 Path Resolution
- **New `resolve_path()` function**: Automatically resolves paths relative to workspace root
- **Applied throughout**: All path-related functions now use `resolve_path()`
- **Benefit**: Can run scripts from any subdirectory with relative paths

### 3.2 nuScenes Token Resolution
- **Fixed "0 samples" issue**: Proper token resolution from pkl files
- **Support for both dataset structures**: Works with `data_list` and `data_infos`
- **Pre-loaded tokens**: Loads tokens directly from pkl for reliability

### 3.3 KITTI Support
- **Complete KITTI benchmark**: Full evaluation pipeline for KITTI
- **Per-class detection counts**: Tracks detections per class
- **Structured JSON output**: `benchmark_kitti.json` with performance metrics

### 3.4 Evaluation Robustness
- **Filtered predictions**: Handles cases where dataset has fewer samples than full validation set
- **Monkey-patching evaluator**: Bypasses assertion errors for filtered predictions
- **Better error handling**: Graceful fallbacks throughout

### 3.5 Output Organization
- **Timestamped directories**: Better organization of results
- **Consistent with MMEngine**: Matches Runner behavior

### 3.6 Environment Setup
- **Numba CUDA enabled**: Fixed CUDA library paths for Numba

---

## 4. Files Affected

### Modified Files:
1. `students/s017530084/simple_infer_main.py`
2. `students/s017530084/simple_infer_utils.py`

### Original Files (for comparison):
1. `simple_infer_main.py` (main directory)
2. `simple_infer_utils.py` (main directory)

---

## 5. Testing & Validation

These changes have been tested with:
- **KITTI Dataset**: PointPillars and Second models
- **nuScenes Dataset**: CenterPoint model
- **Both evaluation backends**: `runner` and `manual`
- **Various path configurations**: Absolute, relative to CWD, relative to workspace root

All changes maintain backward compatibility with the original functionality while adding new features and fixing bugs.

