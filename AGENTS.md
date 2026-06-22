# Repository Guidelines

## Overview

FaceAuth — a real-time face detection and identification system using FaceNet with OpenCV, dlib, and K-means clustering.

## Project Structure & Module Organization

```
camara.py                  # Main entry point: camera UI, registration & identification
face_auth/                 # Core library package
├── facedetector.py        # Dlib Haar-cascade face detection
├── img_processor.py       # Image processing: crop, eye-landmark alignment, histogram equalization
├── face_encoder.py        # TensorFlow FaceNet frozen model (128-dim embeddings)
├── identifier.py          # Encoder + K-means classification orchestrator
├── kmean.py               # Unsupervised clustering with centroid-based classification
├── datas/                 # dlib landmark model (shape_predictor_68_face_landmarks.dat)
├── haarcascade_detectors/ # Haar cascade XMLs (frontalface, eye)
├── models/                # Keras model weights (model.h5)
├── weights/               # InceptionResNetV1 layer weights (.csv)
frozen_models/             # Pre-trained FaceNet frozen_graph.pb
dataset/                   # Generated data — {name}_dataset.csv, {name}_database.csv
requirements.txt           # Python dependencies
important_requirements.txt # Dependency ordering notes
```

## Build, Test & Development Commands

### Setup (required first step)
```bash
pip install -r important_requirements.txt
pip install -r requirements.txt
```
Install in order — `important_requirements.txt` must come first.

### Running
```bash
python camara.py
```
Launches the interactive camera interface.

### Interactive Controls
| Key | Action |
|-----|--------|
| `R` | Register a new user — collects 30 face images, encodes, creates a centroid |
| `I` | Identify faces — real-time identification against known centroids |

### Camera Configuration
Set the camera device in `camara.py`:
```python
device_cam = 1  # Change to 0, -1, or an IP stream URL as needed
```

## Data & Model Paths

- **Registration**: At least 10 images per person for stable centroids (30 recommended).
- **Centroid storage**: `dataset/{name}_database.csv`.
- **FaceNet model**: `./frozen_models/frozen_graph.pb`.

## Coding Style

- **Language**: Python.
- **Indentation**: 4 spaces, no tabs.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Imports**: Grouped standard library → third-party → local (blank-line separated).

## Architecture

### Registration Flow
Capture 30+ images → detect/crop/align/equalize → encode via FaceNet → compute mean → update K-means centroid.

### Identification Flow
Detect → align → encode (128-dim) → classify by nearest centroid → return user ID or `-1`.

## Notes

- FaceNet is pre-trained; no fine-tuning needed for new users.
- K-means centroids grow incrementally as users register.
- Adjust identification sensitivity in `face_auth/identifier.py`.
