# SL-AD

## Detecting Electromagnetic Temporal Anomalies Associated with Major Earthquakes via a Two-Stage Deep Learning Framework

This repository contains the official implementation of **SL-AD**, a two-stage deep learning framework designed for identifying electromagnetic temporal anomalies associated with major earthquakes under highly imbalanced data conditions.

The framework consists of two stages:

1. **Stage I: Sequence Learning**
   - CNN-based local feature extraction
   - Autoformer-based temporal dependency modeling
   - Self-supervised pretraining on large-scale geomagnetic observations

2. **Stage II: Anomaly Identification**
   - Transfer learning from the pretrained temporal encoder
   - Earthquake-related anomaly classification

The method is evaluated using multi-station geomagnetic observations collected from Western China.

---

# Repository Structure

```text
SL-AD/
│
├── backbone/          # Model architectures
├── checkpoint/        # Pretrained checkpoints
├── dataset/           # Dataset files
├── plugin/            # Optional modules
├── util/              # Utility functions
│
├── train_test.py      # Training and evaluation script
├── test.py            # Quick evaluation script
│
└── README.md
```

---

# Dataset

The dataset used in this study is publicly available.

### Dataset Download

Google Drive:

```text
[Dataset Download Link]
```

After downloading, place all files into the `dataset/` directory:

```text
dataset/
├── datasetdown_pca_train.pkl
├── datasetdown_pca_test.pkl
├── xxxx.npy
├── xxxx.npy
└── ...
```

---

# Important: Update Dataset Paths

The files

```text
datasetdown_pca_train.pkl
datasetdown_pca_test.pkl
```

contain file paths used for dataset indexing.

After downloading the dataset, users must update these paths according to their local environment.

Example:

Original path:

```python
/home/author/data/station001.npy
```

Replace with:

```python
/path/to/SL-AD/dataset/station001.npy
```

Failure to update these paths will result in:

```text
FileNotFoundError
```

during data loading.

---

# Pretrained Checkpoint

The pretrained checkpoint should be placed in:

```text
checkpoint/
```

Example:

```text
checkpoint/
└── results_prauc.pth
```

If the checkpoint is not included in this repository due to file size limitations, it can be downloaded from:

```text
[https://drive.google.com/file/d/13qck1xCnZcGxHm_Y0GKkL1EXTeVwoXn7/view?usp=drive_link]
```

---

# Environment

Recommended environment:

```text
Python 3.9
PyTorch 2.0+
CUDA 11.8
NumPy
Pandas
Scikit-learn
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Train the anomaly identification model:

```bash
python train_test.py --is_training 1
```

The best-performing model will be automatically saved according to validation ROC-AUC.

---

# Testing

Evaluate a pretrained model:

```bash
python test.py
```

or

```bash
python train_test.py --is_training 0
```

The following metrics will be reported:

- Accuracy (ACC)
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- False Positive Rate (FPR)
- False Negative Rate (FNR)

---

# Quick Reproduction

After downloading the dataset and pretrained checkpoint, simply run:

```bash
python test.py
```

This command reproduces the main classification results reported in the paper.

---

# License

This project is released under the MIT License.
