# Real-Time Object Detector — Capture, Annotate, Train, Deploy

A general-purpose machine learning pipeline for training and deploying a custom real-time object detector on **any single object class** — no labeling service, no cloud dependency, no pre-existing dataset required. Point a camera at something, collect data, train, and detect.

The system is built around an **active learning loop**: motion-triggered capture builds your dataset automatically, an interactive annotator lets you clean it up, and the live inference tool can feed high-confidence detections back into training data to improve the model over time.

> **Demonstrated on:** steel cans (single object) and stacked pallets of cans in an industrial environment — but the pipeline is object-agnostic.

---

## Motivation

This project was built to add low-cost machine vision to conveyor control systems. Once trained, the model can be consumed by a separate Python script that uses the [pylogix](https://github.com/dmroeder/pylogix) library to write PLC tags based on detection output — for example, indicating which zone a pallet is currently occupying on a conveyor line. This gives a PLC the spatial awareness to make control decisions (stopping a conveyor, triggering a divert, gating a process) without dedicated hardware vision sensors.

This repository covers the **training side** of that system: building and iteratively improving the detection model. The inference output is intentionally simple (a bounding box + confidence score) so it's straightforward to consume from any downstream script or integration.

---

## Example

![Demo](demo.jpg)

*Pallet detection in a factory setting.*

---

## How It Works

```
capture_motion_dataset.py   →   dataset/flat/        (raw frames)
                                dataset/labels/      (bounding boxes as x1 y1 x2 y2)
                                dataset/boxed/       (preview images with drawn boxes)
                                        ↓
box_corrector.py            →   interactive re-annotation of existing samples
                                        ↓
train_torch_detector.py     →   models/frcnn_best.pth
                                        ↓
run_can_detector.py         →   live inference + optional autosave back to dataset
```

Each step feeds the next. Running the detector in autosave mode continuously grows your dataset, and periodic retraining improves accuracy — without any manual labeling effort.

The model only trains from images that exist in the dataset/boxed directory. Before training you can look through this and remove anything you dont want in your trainingset and the respective flat image and labels will be ignored. This creates an easy to use loop to fine tune and guide your training. 

---

## Quickstart

```bash
# 1. Collect initial training data — aim the camera at your object and move it around
python capture_motion_dataset.py

# 2. (Optional) Review and correct bounding boxes
python box_corrector.py

# 3. Train the model
python train_torch_detector.py

# 4. Run the detector live — press 'a' to enable autosave and keep growing your dataset
python run_can_detector.py
```

Repeat steps 1 → 3 as you accumulate more data. The more the detector runs, the better it gets.

---

## Scripts

### `capture_motion_dataset.py` — Motion-Triggered Data Collection

Uses OpenCV's MOG2 background subtractor to detect moving objects and automatically save annotated frames at a configurable rate. Includes black-region suppression because I was using a black handle to move the can I wanted to train on. Remove this if you need to train on a black object.

- Auto-scans for available cameras or uses a preferred index
- Configurable motion sensitivity, minimum contour area, and save cooldown
- `s` — manually save a positive sample
- `n` — save a negative (background-only) sample
- `q` / `ESC` — quit

```bash
python capture_motion_dataset.py
```

---

### `box_corrector.py` — Bounding Box Annotation Tool

An interactive OpenCV-based annotator for reviewing and correcting bounding boxes on existing dataset images. Iterates through every image in `dataset/boxed/` and lets you redraw or clear boxes before training.

| Key | Action |
|-----|--------|
| Drag | Draw new bounding box |
| `s` | Save and advance |
| `x` | Clear current box |
| `n` / `p` | Next / previous image |
| `q` / `ESC` | Quit |

```bash
python box_corrector.py
```

---

### `train_torch_detector.py` — Model Training

Fine-tunes a pretrained Faster R-CNN (ResNet-50 FPN backbone) on your dataset. Handles train/val splitting automatically and saves the best checkpoint by validation loss.

- ImageNet-pretrained backbone weights for fast convergence on small datasets
- 80/20 train/val split, reproducible via seed (splits cached to `dataset/splits/`)
- Horizontal flip augmentation
- AdamW optimizer
- Best model saved to `models/frcnn_best.pth`

```bash
python train_torch_detector.py
```

| Parameter | Default |
|-----------|---------|
| Epochs | 12 |
| Batch size | 2 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Val fraction | 0.20 |

---

### `run_can_detector.py` — Live Inference

Loads the trained model and runs real-time detection on a camera feed. Includes exponential box smoothing and an autosave mode that feeds high-confidence frames back into the training dataset.

- Configurable score and autosave thresholds
- Box smoothing across frames to reduce jitter
- `a` — toggle autosave mode
- `s` — manually save current detection as a positive
- `n` — save current frame as a negative
- `q` / `ESC` — quit

```bash
python run_can_detector.py
```

---

## Dataset Format

```
dataset/
├── flat/        # Raw captured frames (JPEG)
├── labels/      # One .txt per image: "x1 y1 x2 y2", or empty for negatives
├── boxed/       # Preview images with bounding boxes drawn
└── splits/      # Auto-generated train/val split lists
```

Labels use absolute pixel coordinates. An empty label file marks a negative (background-only) sample.

---

## Model

Faster R-CNN with a ResNet-50 FPN backbone, fine-tuned as a two-class detector (background + target object). Only the box predictor head is replaced — the backbone stays pretrained — which makes it practical to train on small, self-collected datasets.

---

## Requirements

```bash
pip install torch torchvision opencv-python numpy Pillow
```

GPU training is recommended but not required. The training script automatically uses CUDA if available.

---

## Project Structure

```
├── capture_motion_dataset.py   # Data collection
├── box_corrector.py            # Annotation tool
├── train_torch_detector.py     # Model training
├── run_can_detector.py         # Live inference + autosave
├── dataset/                    # Created automatically
└── models/                     # Created automatically
    └── frcnn_best.pth
```
