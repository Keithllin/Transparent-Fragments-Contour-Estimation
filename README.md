# Transparent Fragments Contour Estimation via Visual–Tactile Fusion for Autonomous Reassembly

## Automated Batch Generation of Synthetic Dataset in Blender

Blender provides powerful visual simulation capabilities through physically-based rendering (PBR) and material reflection/refraction, producing results that closely mimic real-world interactions between light and materials. 

**Transparent objects**, being a special category, have refractive and transmissive material properties that make their visual features highly sensitive to environmental lighting and background. In real-world scenarios, collecting data of transparent objects with diverse backgrounds and lighting conditions is challenging, and annotations are prone to errors due to difficulties in recognition.

This project offers a Blender script for the **automated generation of synthetic datasets** containing transparent objects. Before each render, the scripts randomize scene elements, including background, lighting and camera angle, greatly enhancing the diversity and richness of the dataset. Using **ID masks**, accurate segmentation masks can be generated. The script supports batch dataset generation with **any scene in which objects are placed at a horizontal plane**. For implementation details, please refer to the [Automated_Batch_Generation_of_Synthetic_Datasets_in_Blender](https://github.com/Keithllin/Transparent-Fragments-Contour-Estimation/tree/main/Automated_Batch_Generation_of_Synthetic_Datasets_in_Blender)
 folder.
### TransFrag27K Dataset

This script was used to create **the first large-scale transparent fragment dataset, TransFrag27K**, which contains **27,000 images and masks** at a resolution of 640×480. 
The dataset covers **fragments of common everyday glassware** and incorporates more than **150 background textures** and **100 HDRI environment lightings**.  

📥 **Download:** The dataset [TransFrag27K on Hugging Face](https://huggingface.co/datasets/chenbr7/TransFrag27K) is available to download. 

## Mask Segmentation and Corner Extraction for Transparent Object Fragments Transparent objects
Robust Mask Segmentation and Corner Extraction for Transparent Object Fragments Transparent objects exhibit strong refraction and transmission, making their visual appearance highly sensitive to lighting and background. Collecting diverse, well-annotated real-world data is difficult, and segmentation of transparent fragments is especially error-prone yet crucial for downstream manipulation tasks.

This project provides a Swin Transformer–based segmentation pipeline that predicts high-quality binary masks from RGB images and derives corner endpoints along dominant edges via post-processing. The system includes careful preprocessing, illumination-robust normalization, optional test-time augmentation (TTA), and brightness/contrast/gamma enhancements to improve performance under challenging lighting. Evaluation metrics (IoU, Dice, pixel accuracy), visualization, and best-checkpoint selection are integrated.

The codebase offers an end-to-end training loop with checkpointing and logging, as well as a batch inference script that outputs masks, corner coordinates, and qualitative overlays. It supports datasets organized as PNG images with corresponding NumPy mask files and is configurable through a single settings module. For implementation details, please refer to train.py, inference.py, and config/settings.py.

## Overview

This repository implements a semantic segmentation pipeline to predict object masks from RGB images and optionally derive corner endpoints for the longest edge within each detected instance via post-processing. The core model uses a timm backbone (default: Swin Transformer) and a lightweight FPN/UNet-style decoder. The project provides a full training loop with checkpointing, logging, visualization, and an inference script that supports test-time augmentation (TTA), brightness enhancement, mask smoothing, and multi-object corner extraction.

Primary code entry points:

- `train.py`: end-to-end training and validation.
- `inference.py`: batch inference on a folder of images, with optional TTA and post-processing.
- `config/settings.py`: all configuration (paths, hyperparameters, augmentations, logging).


## Key Features

- Backbone from `timm` (default `swin_base_patch4_window7_224.ms_in22k_ft_in1k`) with `features_only=True`.
- Improved decoder with lateral connections and progressive upsampling; output is a single-channel mask logit map.
- Combined loss configured for mask segmentation; evaluation metrics include IoU, Dice, and pixel accuracy.
- Robust training utilities: checkpointing (latest/best/per-epoch), learning rate scheduling, gradient clipping, WandB logging (defaults to offline mode), and periodic visualizations.
- Inference utilities: optional TTA, brightness/contrast/gamma enhancement, mask smoothing (Gaussian, morphology, median), and corner extraction from predicted masks.


## Repository Structure (abridged)

```text
Tactile-Vision/
	train.py                  # Training entry point
	inference.py              # Inference entry point
	config/
		settings.py             # All configuration
	data/
		dataset.py              # Dataset and IO
		transforms.py           # Preprocessing & augmentation
	model/
		network.py              # Model definition (timm backbone + decoder)
	utils/
		losses.py, metrics.py, postprocessing.py, visualize.py
	outputs/<EXPERIMENT>/     # Checkpoints, logs, visualizations, inference results
```


## Requirements

Tested with Python 3.9+ on Windows and Linux. Recommended packages:

- torch, torchvision (CUDA optional)
- timm
- albumentations
- opencv-python
- numpy, Pillow, matplotlib
- tqdm
- wandb

You can install typical dependencies via pip (adjust CUDA/Torch install to your platform and GPU driver):

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm albumentations opencv-python numpy pillow matplotlib tqdm wandb
```


## Dataset Preparation

Set `DATASET_ROOT_DIR` in `config/settings.py` to the dataset root. The expected directory layout is:

```text
<DATASET_ROOT_DIR>/
	train/
		rgb/          # training images (PNG)
		anno_mask/    # training masks (.npy, binary/float arrays)
		corner/       # optional corner annotations (.npy), not required for training
	test/
		rgb/
		anno_mask/
		corner/
```

File naming conventions:

- Image files are `*.png` under `rgb/` (e.g., `1_Color.png` or `1.png`).
- Mask files are `*.npy` under `anno_mask/`. For an image `1_Color.png`, the loader will look for `1.npy` (the `_Color` suffix is removed).

The dataset loader `CornerPointDataset` uses only images and masks for training; corner arrays are optional and only used if explicitly enabled. Images and masks are transformed to a fixed resolution given by `IMAGE_HEIGHT` and `IMAGE_WIDTH` in `settings.py`.


## Configuration

All settings live in `config/settings.py`. Key options:

- General: `PROJECT_NAME`, `EXPERIMENT_NAME`, `DEBUG_MODE`, `RANDOM_SEED`, `DEVICE`.
- Data: `DATASET_ROOT_DIR`, `TRAIN_DIR`, `TEST_DIR`, `VAL_SPLIT_RATIO`, `NUM_WORKERS`, `PIN_MEMORY`.
- Model: `MODEL_NAME`, `PRETRAINED_BACKBONE`, `PRETRAINED_MODEL_PATH`, `IMAGE_HEIGHT`, `IMAGE_WIDTH`.
- Optimization: `BATCH_SIZE`, `LEARNING_RATE`, `WEIGHT_DECAY`, `OPTIMIZER_TYPE`, `SCHEDULER_TYPE`, `GRAD_CLIP_MAX_NORM`, `NUM_EPOCHS`.
- Loss/Metrics: `MASK_LOSS_WEIGHT`.
- Logging and outputs: `BASE_OUTPUT_DIR`, `MODEL_SAVE_DIR`, `LOG_DIR`, `VISUALIZATION_DIR`, `LOG_FILE_NAME`.
- Checkpointing/validation: `SAVE_MODEL_EVERY_N_EPOCHS`, `VALIDATE_EVERY_N_EPOCHS`, `SAVE_BEST_MODEL_ONLY`, `BEST_MODEL_METRIC`, `BEST_MODEL_METRIC_MODE`, `RESUME_TRAINING`, `CHECKPOINT_PATH`.
- Visualization: `VISUALIZE_PREDICTIONS_EVERY_N_EPOCHS`, `NUM_VALIDATION_PREDICTIONS_TO_VISUALIZE`, `WANDB_*` options. Note train.py sets `WANDB_MODE=offline` by default.
- Inference-specific: `INFERENCE_MODEL_PATH`, `INFERENCE_INPUT_DIR`, `INFERENCE_OUTPUT_DIR`, `INFERENCE_IMG_EXTENSIONS`, `USE_TEST_TIME_AUGMENTATION`, `TTA_SCALES`, `TTA_FLIPS`, `INFERENCE_BRIGHTNESS_ENHANCEMENT`, `INFERENCE_BRIGHTNESS_FACTOR`, `INFERENCE_CONTRAST_FACTOR`, `INFERENCE_GAMMA_CORRECTION`, `INFERENCE_MAX_OBJECTS`, `INFERENCE_MIN_CONTOUR_AREA`.
- Mask smoothing presets: `MASK_SMOOTH_EDGES`, `MASK_SMOOTH_TYPE`, `MASK_SMOOTH_PRESETS` and low-level toggles.

Adjust these values before running training or inference.


## Training

1) Set dataset path and training options in `config/settings.py`.

2) Start training:

```bat
.venv\Scripts\activate
python train.py
```

During training:

- Checkpoints are saved under `outputs/<EXPERIMENT_NAME>/checkpoints/` as `latest_model.pth`, `best_model.pth` (based on `BEST_MODEL_METRIC`), and optional per-epoch files.
- Logs are written to console and `outputs/<EXPERIMENT_NAME>/logs/training_session.log`.
- Validation visualizations (predicted masks and optional corner overlays) are saved in `outputs/<EXPERIMENT_NAME>/visualizations/` at the configured frequency.


To resume training, set `RESUME_TRAINING=True` and ensure `CHECKPOINT_PATH` or the default `latest_model.pth` exists.


## Inference

Configure the following in `config/settings.py`:

- `INFERENCE_MODEL_PATH`: path to a trained checkpoint (e.g., `outputs\<EXPERIMENT_NAME>\checkpoints\best_model.pth`).
- `INFERENCE_INPUT_DIR`: directory containing input images.
- `INFERENCE_OUTPUT_DIR`: directory where inference results will be written.
- Optional: enable `USE_TEST_TIME_AUGMENTATION`, `INFERENCE_BRIGHTNESS_ENHANCEMENT`, and mask smoothing options per your scenario.


Run inference:

```bat
.venv\Scripts\activate
python inference.py
```

Outputs per image are saved to `INFERENCE_OUTPUT_DIR`:

- Visualization overlay `vis_<imageName>.png` (or `tta_vis_*.png` when TTA is used).
- Binary mask `mask_<imageStem>.png` (or `tta_mask_*.png`).
- Corner text file `corners_<imageStem>.txt` listing one line per detected object as `x1,y1,x2,y2` in the original image scale (or `tta_corners_*.txt`).
- If a ground-truth `.npy` mask with a matching base name is found, an `iou_<imageStem>.txt` file is also emitted.



## Evaluation Metrics

The validation loop computes:

- Intersection over Union (IoU)
- Dice coefficient
- Pixel accuracy


Metrics are averaged over the validation set and included in the checkpoint metadata for reference.


## Model Architecture

- Backbone: configurable via `timm` (default Swin Transformer). `features_only=True` is used to obtain multi-scale features.
- Decoder: lateral 1x1 convolutions to align channel dimensions, followed by upsampling and summation (FPN-like). A final convolutional head predicts a single-channel mask logit map.
- Output size: the predicted mask is bilinearly resized to match the input spatial resolution.


## Test-Time Augmentation, Brightness Enhancement, and Mask Smoothing

- TTA: horizontal flip, small rotations, mild scale jitter, and optional brightness adjustments, with prediction fusion by averaging.
- Brightness enhancement: linear brightness/contrast adjustments and gamma correction to improve performance in dark scenes.
- Mask smoothing: Gaussian blur, morphology (open/close), and median filtering to reduce jagged edges. You can select a preset (`gentle`, `moderate`, `aggressive`) or control each operation explicitly.


## Reproducibility

The training script seeds Python, NumPy, and PyTorch RNGs via `set_seed`. For CUDA, it sets `cudnn.deterministic=True` and disables benchmarking to reduce nondeterminism. Full determinism is not guaranteed across all platforms and kernels, but variance is reduced.


## Logs and Outputs

Default output base is `outputs/` under the project root, with subfolders per experiment name:

```text
outputs/ <EXPERIMENT_NAME>/
	checkpoints/     # latest_model.pth, best_model.pth, optional per-epoch
	logs/            # training_session.log
	visualizations/  # training/validation visualizations
	inference_results/
```


## Troubleshooting

- FileNotFoundError for dataset paths: verify `DATASET_ROOT_DIR` and that `train/test` contain `rgb/` and `anno_mask/` folders. Ensure masks are `.npy` and follow the naming rule (`_Color` removed when mapping images to masks).
- Inference exits with a placeholder path error: set `INFERENCE_MODEL_PATH` and `INFERENCE_INPUT_DIR` to real paths in `config/settings.py`.
- Torch or CUDA installation issues: install a Torch build matching your CUDA driver. See the official PyTorch installation guide.
- Empty or poor masks: inspect `IMAGE_HEIGHT/IMAGE_WIDTH`, normalization mean/std, augmentation intensity, and consider enabling brightness enhancement or mask smoothing.
- Corner detection returns NaNs: adjust `INFERENCE_MIN_CONTOUR_AREA`, thresholding, or smoothing presets; ensure objects are large enough and predicted masks are not empty.




## Acknowledgments

This project uses the `timm` model library and common computer vision tooling from the PyTorch ecosystem. Thanks to the open-source community for making these components available.
