# LeafScan

A desktop plant-disease scanner. Point a webcam at a leaf (or import a photo). LeafScan finds the
leaf, classifies it with a fine-tuned **MobileNetV3-Small**, draws an outline and label on the image,
and shows guidance cards with sources. Each scan can be saved to a per-plant history with charts.

> **Guidance, not a diagnosis.** The model was trained on clean lab-style photos. It can be wrong,
> especially on real-world photos. Confirm with a local extension service before treating plants.

Built with PySide6 (Qt), OpenCV, PyTorch and SQLite. Requires Python 3.11+. Tested on Windows 11.

## Features

- **Scan:** live webcam feed (threaded, the UI never freezes), capture (Space), import (Ctrl+O).
- **Pipeline:** capture -> leaf detection -> classification -> overlay -> save. Analysis runs on a
  `QThread` worker; the UI thread never runs inference.
- **Result panel:** condition, confidence bar, alternative guesses, and a **"Not sure"** state when
  confidence is below the threshold (no forced label). Also an off-colour (discoloration) area
  readout with an optional pink highlight, guidance cards with sources, and the disclaimer.
- **No leaf detected:** shown with the reason (too small, not plant-coloured, cluttered background, ...).
- **History:** per-plant filter, summary tiles, a "scans over time" chart, a "conditions per plant"
  chart (both with hover tooltips), a scan table with thumbnails, a detail view and delete.
- **Plants:** pick an existing plant or type a new name when saving. "Plant 1, 2, ..." is suggested.
- **Dark/light theme** (remembered between runs).
- **Hardening:** works without a camera (with a retry button) and without a model (falls back to a
  mock with a visible banner). Bad images are rejected with a message. Unexpected errors go to
  `data/app/leafscan.log`.

## Install

```bash
python -m venv .venv
```
```bash
.venv\Scripts\python -m pip install -r requirements.txt
```
```bash
.venv\Scripts\python -m pip install -e . --no-deps
```

For GPU training, install the CUDA build of PyTorch first (for example
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`).
The app itself runs inference on the CPU, about 10-20 ms per image.

## Run the demo

The trained model (`models/leafscan_mnv3s.pt`, about 6 MB) is committed, so the app works right after install:

```bash
.venv\Scripts\python -m app
```

Useful flags: `--no-camera`, `--camera 1`, `--image leaf.jpg` (analyse at startup), `--theme light`,
`--threshold 0.7` (the "not sure" cut-off, default 0.60; also `LEAFSCAN_THRESHOLD`),
`--data-dir PATH` (where history is stored; default `data/app/`), and `--mock` (no model).

**Demo script (about 3 minutes)**
1. Seed a demo history: `python scripts/seed_demo_history.py`, then start `python -m app --data-dir data/demo`.
2. **Scan:** hold a single leaf against a plain sheet of paper and press **Space**. Point out the
   outline, the label, the confidence and the guidance card with sources. Toggle
   *Highlight discoloration*.
3. Show **"Not sure"**: a blurry or odd photo gives "Not sure" with best guesses instead of a forced label.
4. Show **"No leaf detected"**: capture an empty desk or your hand.
5. **Save** the scan to "Plant 1" (suggested automatically) or a new name.
6. **History:** filter by plant, hover over the charts, open a scan, switch to light theme.

No webcam? Use **Import image** with photos from `data/processed/test/<class>/`.
If you use a phone as a webcam (Iriun, DroidCam, ...), start its app, then press **Retry camera**.

## Classes

Nine classes from **PlantVillage** ([BrandonFors/Plant-Diseases-PlantVillage-Dataset](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset), CC0-1.0).
They are defined only in [`src/app/labels.json`](src/app/labels.json), the single source of truth.
A test fails if a class name appears anywhere in the code.

| id | key | plant | condition |
|---|---|---|---|
| 0 | pepper_bacterial_spot | Pepper | Bacterial spot |
| 1 | pepper_healthy | Pepper | Healthy |
| 2 | potato_early_blight | Potato | Early blight |
| 3 | potato_late_blight | Potato | Late blight |
| 4 | potato_healthy | Potato | Healthy |
| 5 | tomato_bacterial_spot | Tomato | Bacterial spot |
| 6 | tomato_early_blight | Tomato | Early blight |
| 7 | tomato_late_blight | Tomato | Late blight |
| 8 | tomato_healthy | Tomato | Healthy |

## Reproduce the model

```bash
python scripts/download_dataset.py
```
```bash
python scripts/prepare_dataset.py
```
```bash
python scripts/train.py
```
```bash
python scripts/evaluate.py
```

1. `download_dataset.py` fetches about 850 MB of parquet shards in parallel chunks, retrying stalled connections.
2. `prepare_dataset.py` keeps the 9 classes, removes exact duplicates (by SHA-1), and makes a
   **fixed, seeded (42), stratified 70/15/15 split**. The split depends only on image hashes and the
   seed. Counts and a manifest hash are in `reports/dataset_split.json`.
3. `train.py` fine-tunes ImageNet-pretrained MobileNetV3-Small (AdamW, cosine LR with warm-up,
   label smoothing 0.05, sqrt-inverse-frequency class weights). It uses augmentation aimed at the
   webcam gap: rotation, crop/zoom, flips, colour jitter, blur, JPEG artefacts and erasing. It keeps
   the best epoch by validation macro-F1, then fits a softmax **temperature** on the validation set so
   the "not sure" threshold works on calibrated confidences. The test split is evaluated once at the end.
   Weights are saved with a version string (`mnv3s-9c-YYYYMMDD-HHMM`) and metadata
   (`models/leafscan_mnv3s.json`).
4. Reports go to `reports/`: confusion matrices (`*_confusion_matrix.png`), per-class
   precision/recall/F1 (`*_metrics.json`, `*_classification_report.txt`) and training curves.

### Real webcam photos

Put your own photos in `data/real_test/<class_key>/` and run:

```bash
python scripts/evaluate.py --real
```

This runs the **full app pipeline** (detect -> crop -> classify) and writes `reports/real_test_*`.
Whole-image accuracy is reported alongside. Dataset images are much cleaner than webcam shots, so
this is the number to trust.

## Results

Model `mnv3s-9c-20261014-1638` (best epoch 14 of 20, about 15 min of training on a GTX 1650 Ti, 6.2 MB).

| Evaluation | Images | Accuracy | Macro-F1 | "Not sure" (<60%) |
|---|---|---|---|---|
| Validation split | 1,686 | 99.70% | 0.997 | - |
| **Test split** (held out, evaluated once) | 1,686 | **99.82%** | **0.996** | 0.18% |
| Test split through the full app pipeline (detect -> crop -> classify) | 1,686 | 99.76% | - | 0.18% |
| **Real webcam photos** (`data/real_test/`) | 0 | *not measured yet* | | |

- Only 3 of the 1,686 test images are misclassified (see `reports/test_confusion_matrix.png`). The weakest
  class is *Potato - Healthy* (precision 0.958). It has only 152 images in the whole dataset.
- **Leaf detection** finds the leaf in 1,683/1,686 test images (99.82%), about 87 ms each
  (`reports/leaf_detection.json`). The 3 misses are a shrivelled leaf, a close-up running off the
  frame, and a leaf too small in the frame. A photo of a road with trees is correctly rejected.
- The end-to-end scan (detection + inference + overlay) takes about 50-300 ms on a laptop CPU.

**Read the dataset numbers with care.** PlantVillage photos are lab-style: one detached leaf, uniform
background, consistent lighting, and often several photos of the same leaf. Near-perfect test scores
are typical for this dataset and do **not** predict field accuracy. Published work reports large drops
on real-world photos. Add your own photos to `data/real_test/` and run `evaluate.py --real`; that is
the number to quote.

## Architecture

```
src/app/
  contracts.py    dataclasses + Protocols for every module interface, plus mock implementations
  labels.json     class ids / names - single source of truth (labels.py loads it)
  config.py       settings (paths, threshold), LEAFSCAN_* env overrides
  capture.py      webcam QThread + image import (the ONLY BGR->RGB conversion)
  pipeline.py     detect -> crop -> classify -> overlay -> guidance
  main.py         entry point, pipeline factory with model fallback, logging
  ui/             main window shell, camera view, result panel, history view, charts, theme, worker
  classify/       model + checkpoint I/O, preprocessing, predictor, training data/augmentation, metrics
  vision/         classical OpenCV leaf mask (HSV/Lab + GrabCut), discoloration, overlay rendering
  storage/        SQLite schema, PNG image saving, history queries, plant IDs
  guidance/       content.json - per-class guidance with sources
scripts/          download / prepare / train / evaluate / seed demo history
tests/            51 tests: contracts, storage, vision, guidance, hardening, headless UI end-to-end
```

Interfaces (all images are RGB `uint8` numpy arrays):

- `classify.predict(rgb) -> ClassifierResult(class_id, disease_name, confidence, is_uncertain, ...)`
- `vision.detect_leaf(rgb) -> LeafResult(mask, annotated_image, found, bbox, ...)`, in input coordinates
- `storage.save_scan(ScanRecord(image, plant_id, timestamp, prediction, confidence, model_version, ...))`

## Known limitations

- **Domain gap.** Trained only on PlantVillage lab photos. Expect lower accuracy on webcam photos with
  different lighting, attached leaves, wet or dirty leaves, or other cultivars. Real-photo accuracy has
  not been measured yet (`data/real_test/` is empty).
- **Closed set of 9 classes.** Other diseases, pests, nutrient deficiencies or other plant species are
  forced into one of the 9 classes or reported as "Not sure". The model cannot say "unknown disease".
- **Overconfidence.** Calibration (temperature 0.53) is fitted on clean validation images, so
  confidences on unusual photos are still too high. The 60% "not sure" threshold is a safeguard, not a
  guarantee. Confidence is never shown as 100%.
- **Leaf detection needs a plain background.** Leaves among foliage, held in a hand against a busy
  scene, or running off the frame edges are rejected with "No leaf detected" and a reason. One leaf per
  scan.
- **Discoloration highlight** marks off-colour tissue (lesions, yellowing, but also pale veins or glare).
  It is a visual aid, not a severity score.
- **Possible near-duplicates.** Exact duplicates are removed before splitting, but PlantVillage contains
  multiple photos of the same physical leaf. Some may sit in both train and test, which inflates test
  scores.
- **Guidance** is general home-garden information with sources. It is not region-specific and not a
  substitute for local extension advice or pesticide labels.
- Tested on Windows 11 with Python 3.11. macOS/Linux should work (no Windows-only code paths apart
  from the camera backend choice) but are untested.

## Tests

```bash
.venv\Scripts\python -m pytest
```

UI tests run headless (`QT_QPA_PLATFORM=offscreen`).

## Credits

- Dataset: PlantVillage (Hughes & Salathe), via the Hugging Face mirror above (CC0-1.0).
- Guidance content was written for this project and links to UC IPM, NC State Extension, Cornell
  and RHS pages.
- Inspired by [manthan89-py/Plant-Disease-Detection](https://github.com/manthan89-py/Plant-Disease-Detection)
  (ideas only; no code reused).
