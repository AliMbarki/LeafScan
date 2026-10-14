# LeafScan

Desktop plant-disease scanner (PySide6 + OpenCV + PyTorch). Work in progress - see phases below.

## Quick start (Windows, Python 3.11+)

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pip install -e . --no-deps
    .venv\Scripts\python -m app            # or: --no-camera, --image path.jpg, --theme light

Run tests: `.venv\Scripts\python -m pytest`

## Status
- [x] Phase 1 - skeleton: contracts + mocks, UI shell, SQLite history
- [ ] Phase 2 - MobileNetV3-Small classifier
- [ ] Phase 3 - leaf detection + overlays
- [ ] Phase 4 - history charts + guidance
- [ ] Phase 5 - hardening, docs
