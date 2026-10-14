# Real webcam test photos

Put real photos here, one sub-folder per class **key** from `src/app/labels.json`
(e.g. `tomato_early_blight/IMG_001.jpg`). Then run:

    python scripts/evaluate.py --real

Accuracy on these photos is reported separately from the dataset test split
(`reports/real_test_metrics.json`).
