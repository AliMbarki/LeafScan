# LeafScan

A light desktop plant-disease scanner. Point a webcam at a leaf (or import a photo): LeafScan finds the leaf,
identifies the plant and its condition with a **pre-trained model** (no training needed), outlines the leaf and
shows guidance with sources. Scans can be saved to a per-plant history with charts.

> **Guidance, not a diagnosis.** Confirm with a local extension service before treating plants.

PySide6 (Qt) + OpenCV + ONNX Runtime + SQLite. Python 3.11+. No PyTorch, no GPU needed.

## Install and run

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python -m pip install -e . --no-deps
```
```bash
.venv\Scripts\python -m app
```

Options: `--no-camera`, `--camera 1`, `--image leaf.jpg`, `--theme light`, `--data-dir PATH`,
`--threshold 0.6` (condition), `--plant-threshold 0.55` (plant).

Keys: **Space** capture, **Ctrl+O** import, **Ctrl+S** save, **Esc** back to live.
Tip: one leaf in the middle of the frame, even light; a plain background gives the cleanest outline.

## Model

[`models/plant_disease.onnx`](models/plant_disease.onnx) (9 MB): MobileNetV2 fine-tuned on PlantVillage by
[linkanjarad](https://huggingface.co/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification), ONNX export by
onnx-community (licence: "other", see the model card). 38 classes, **14 plants**: apple, blueberry, cherry, corn,
grape, orange (citrus), peach, pepper, potato, raspberry, soybean, squash, strawberry, tomato.
Classes and plant names live only in [`src/app/labels.json`](src/app/labels.json).

Measured on held-out photos of those 14 plants:

| Photos | Plant correct | Condition correct |
|---|---|---|
| Lab photos (PlantVillage style, plain background) | 97% | 94% |
| Real-world photos (PlantDoc / field datasets) | 64% | 24% |

The model was trained on lab photos only, so real-world webcam photos are much harder. On those its
confidence is usually low, so the app answers "Not sure" instead of guessing. Leaves of other plants are
not recognised.

## How it works

`capture -> leaf detection -> classification -> overlay -> save`, with analysis on a background thread.

- **Leaf detection** (OpenCV): isolated leaf on a plain background, otherwise GrabCut on the object in the
  middle of the frame, otherwise the main green area. "No leaf detected" only when no plant is visible.
- **Classification**: the leaf crop, its mirror image and (on busy scenes) the whole frame are averaged.
  The plant is decided separately (sum of its classes), so you can get "Tomato - condition not sure".
- **Guidance**: [`src/app/guidance/content.json`](src/app/guidance/content.json), cards per condition with
  links to UC IPM, RHS, NC State, Ohio State, Wisconsin, Cornell and UF/IFAS extension pages.

```
src/app/
  contracts.py  labels.json/labels.py  config.py  capture.py  pipeline.py  main.py
  classify/   ONNX predictor + preprocessing
  vision/     leaf detection + overlay
  storage/    SQLite history
  guidance/   guidance content
  ui/         main window, camera view, result panel, history + charts, supported-plants dialog
models/       plant_disease.onnx + plant_disease.json (preprocessing, version)
tests/
```

Run tests: `.venv\Scripts\python -m pytest`
