# YOLOv8-Based Object Detection in UAV Imagery

**Adaptive Region-Based Inference for Small-Object Detection**

## Overview

This project investigates object detection in aerial imagery using YOLOv8n and the VisDrone2019-DET dataset. The main objective is to establish a baseline detector, investigate fine-tuning and higher-resolution inference, and evaluate an adaptive region-based approach for detecting objects in crowded aerial scenes.

Small objects, dense object distributions, and limited image resolution make aerial object detection challenging. This project explores whether selectively processing image regions can complement full-image object detection.

## Objectives

- Train and evaluate a YOLOv8n baseline on VisDrone.
- Analyze detection performance across inference resolutions.
- Investigate fine-tuning as a way to improve baseline performance.
- Develop an adaptive region-selection and crop-inference pipeline.
- Train a regional specialist detector using cropped training images.
- Compare full-image and regional detection configurations on the original validation set.

## Dataset

**VisDrone2019-DET** is an aerial object detection dataset containing images captured from drones in diverse urban and outdoor environments.

The project uses the training and validation splits. The original validation images and annotations are retained for evaluation of the detection pipelines.

Download the dataset from the official VisDrone website:

https://aiskyeye.com/

After downloading and preparing the dataset, place it in the following directory structure:

```text
datasets/
└── VisDrone/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── VisDrone.yaml
```

The directory layout may need to be adjusted to match the dataset configuration used by the notebook.

**Note:** The dataset is not included in this repository. Follow the dataset provider's terms and instructions when downloading and using it.

## Methodology

### 1. Baseline Detection

A pretrained YOLOv8n model is fine-tuned on the VisDrone training split. The initial baseline is trained for up to 20 epochs at an image size of 640 pixels.

The baseline provides a reference for subsequent experiments.

### 2. Error and Resolution Analysis

The baseline is evaluated using validation metrics, training curves, confusion analysis, and visual comparisons between predictions and ground-truth annotations.

Higher-resolution inference is also investigated to assess its effect on detecting objects of different sizes.

### 3. Full-Image Fine-Tuning

A second model is fine-tuned from the baseline checkpoint. Its performance is compared with the original baseline under the same validation conditions.

### 4. Adaptive Region-Based Inference

An adaptive region-selection pipeline identifies image regions for additional processing. The selection strategy considers object size, prediction confidence, and detection distribution.

Selected regions are cropped and processed separately. The resulting detections are combined with full-image predictions, with duplicate detections suppressed during prediction fusion.

### 5. Regional Specialist

A regional specialist is trained using 480 × 480 crops generated from the training images. Ground-truth bounding boxes are transformed into the corresponding crop coordinates.

The specialist is evaluated on selected regions from the original validation images. The original validation set remains separate from crop generation for training.

### 6. Comparative Evaluation

The experiments compare baseline detection, full-image fine-tuning, and regional detection configurations on the same 548-image validation set.

Additional experiments investigate confidence thresholds and detection performance across object-size groups.

## Experimental Results

### Full-Image Detection

The baseline model was evaluated at multiple inference resolutions.

| Inference image size | mAP@50 | mAP@50–95 |
|---:|---:|---:|
| 640 | 0.2686 | 0.1510 |
| 800 | 0.3113 | 0.1762 |
| 960 | 0.3353 | 0.1931 |
| 1024 | 0.3448 | 0.1984 |
| 1280 | 0.3523 | 0.2027 |

Higher-resolution inference improved the measured baseline performance in these experiments.

### Baseline vs. Fine-Tuned Model vs. Regional Specialist

The following results were obtained from direct evaluation at an image size of 640 on the 548-image validation set.

| Model | mAP@50 | mAP@50–95 | Precision | Recall |
|---|---:|---:|---:|---:|
| A — Baseline | 0.2687 | 0.1510 | 0.3961 | 0.2970 |
| B — Fine-tuned | 0.2738 | 0.1537 | 0.4042 | 0.3023 |
| C — Regional specialist | 0.2362 | 0.1307 | 0.3860 | 0.2686 |

The fine-tuned model achieved modest improvements over the baseline in this evaluation. The regional specialist performed worse when evaluated directly on full images at the same resolution, so its value must be assessed within the regional inference pipeline rather than assumed from standalone performance.

### Regional Confidence-Threshold Analysis

A separate experiment compared the baseline detector and regional specialist as crop detectors, using the same selected regions and fixed full-image predictions.

At a final confidence threshold of 0.50, the custom overall F1 scores were approximately:

| Crop detector | Overall F1 |
|---|---:|
| Baseline detector | 0.3796 |
| Regional specialist | 0.3799 |

The difference was marginal. Results varied with the confidence threshold and object-size group.

These F1 scores come from a custom matching and evaluation procedure. They are not COCO mAP scores and should not be interpreted as directly comparable to the mAP results above.

**Interpretation:** The experiments demonstrate the implementation and evaluation of an adaptive region-based detection pipeline, but the current results do not establish a substantial performance improvement over the baseline.

## Technologies

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- CUDA-enabled GPU acceleration

## Repository Structure

```text
visdrone-yolov8-object-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── visdrone_yolov8_regional_detection.ipynb
└── results/
    ├── regional_confidence_threshold_comparison.xlsx
    └── regional_confidence_threshold_sweep.xlsx
```

The filenames shown above are examples. The repository should contain the actual final notebook and selected result files.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA environment if GPU acceleration is required. Refer to the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

Download and prepare the VisDrone dataset, then update the dataset and project paths in the notebook to match your local directory structure.

Open the notebook using Jupyter:

```bash
jupyter notebook
```

**Reproducibility:** The notebook currently relies on local dataset files and trained checkpoints generated during the experiments. A fresh run requires preparing the dataset and recreating the required checkpoints in the appropriate order.

## Limitations

- Small and densely packed objects remain challenging to detect.
- The regional specialist does not outperform the baseline in every evaluation setting.
- The observed improvements in the custom regional F1 comparison are marginal.
- Some regional experiments use custom matching and size-group diagnostics rather than standard COCO evaluation.
- Exact reproduction requires the appropriate dataset, model checkpoints, dependencies, and configuration paths.

## Future Work

- Improve adaptive region selection and crop coverage.
- Investigate more effective fusion of full-image and regional predictions.
- Evaluate alternative confidence thresholds and duplicate-suppression strategies.
- Perform standardized COCO evaluation and per-class analysis.
- Investigate inference-time and memory trade-offs for higher-resolution detection.

## Acknowledgements

This project uses the VisDrone dataset and the Ultralytics YOLO implementation.

- VisDrone: https://aiskyeye.com/
- Ultralytics documentation: https://docs.ultralytics.com/

## License

Add a license only after confirming that the licensing terms of the code, dependencies, and any included materials permit your intended distribution.
