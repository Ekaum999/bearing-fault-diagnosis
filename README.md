# Bearing Fault Detection — Paderborn University Dataset

A deep learning system that automatically detects and classifies bearing faults
from raw vibration signals, achieving **97.13% accuracy** using a Conv-BiLSTM
architecture with attention mechanism.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 97.13% ± 0.31% |
| F1 Score | 0.9713 ± 0.0031 |
| ROC AUC | 0.9973 ± 0.0006 |

Evaluated using **5-fold stratified cross validation** on 30,406 signal segments.

---

## Problem Statement

Bearing faults are one of the most common causes of industrial machine failure.
Early detection prevents costly breakdowns and unplanned downtime. This project
builds a model that listens to raw vibration signals and automatically identifies
what type of fault — if any — is present.

---

## Dataset

**Paderborn University Bearing Dataset**
- Sampling rate: 64,000 Hz
- Signal length per segment: 12,800 samples (0.2 seconds)
- 4 classes:

| Class | Description | Segments |
|-------|-------------|----------|
| Healthy | No fault | 8,001 |
| IR | Inner race fault | 9,612 |
| OR | Outer race fault | 7,994 |
| IR & OR | Combined inner + outer race fault | 4,799 |

Dataset available at: https://mb.uni-paderborn.de/kat/forschung/kat-datacenter/bearing-datacenter/data-sets-and-download

---

## Methodology

### Signal Processing Pipeline

```
Raw vibration signal (64,000 Hz)
        ↓
High-pass filter at 1,000 Hz
(removes low frequency noise)
        ↓
Segment into 12,800 sample windows
        ↓
Per-sample normalization
        ↓
Conv-BiLSTM model
        ↓
Fault classification (4 classes)
```

### Model Architecture — Conv-BiLSTM with Attention

```
Input: (batch, 1, 12800)
        ↓
Conv1D (kernel=64, stride=8)  → local feature extraction
        ↓
Conv1D (kernel=16, stride=4)  → higher level features
        ↓
BiLSTM (hidden=128)           → temporal patterns (forward + backward)
        ↓
BiLSTM (hidden=64)            → deeper temporal patterns
        ↓
Attention layer               → focus on most informative time steps
        ↓
Fully connected classifier
        ↓
Output: 4 class probabilities
```

### Why BiLSTM?

A standard LSTM reads the signal only forward in time. A **Bidirectional LSTM**
reads it both forward AND backward simultaneously, capturing patterns that only
become clear when looking at the signal from both directions — particularly
useful for impact-type signals like bearing faults.

### Why Attention?

Not all time steps in a vibration signal are equally informative. The attention
mechanism learns to focus on the moments where fault impacts actually occur,
improving both accuracy and interpretability.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss function | Cross Entropy |
| Epochs | 50 |
| Batch size | 128 |
| Regularization | Dropout (0.3) + Gradient clipping |
| LR scheduler | ReduceLROnPlateau (patience=5) |
| Validation split | 10% of training data |
| Cross validation | 5-fold stratified |
| Hardware | NVIDIA RTX 4060 Laptop GPU |

---

## Repository Structure

```
bearing-fault-detection-paderborn/
│
├── notebooks/
│   └── Conv_BiLSTM.ipynb       ← main experiment notebook
│
├── src/
│   └── loader/
│       └── load_dataset.py     ← data loading and preprocessing
│
├── results/
│   └── confusion_matrices/     ← per-fold confusion matrix plots
│
├── data/
│   └── README.md               ← dataset download instructions
│
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Ekaum999/bearing-fault-detection-paderborn.git
cd bearing-fault-detection-paderborn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Follow instructions in `data/README.md` to download the Paderborn dataset.
Place the extracted folders in `C:\dataset\` (or update the path in the notebook).

### 4. Run the notebook
Open `notebooks/Conv_BiLSTM.ipynb` in VS Code or Jupyter and run all cells.

---

## Requirements

```
torch
numpy
scipy
scikit-learn
matplotlib
seaborn
```

---

## Key Takeaways

- Raw vibration signals can be classified directly without hand-crafted features
- Bidirectional LSTMs capture temporal dependencies in both directions
- Attention mechanism improves focus on fault impact moments
- 5-fold cross validation ensures results are reliable and not overfitted
- Combined IR&OR fault class shows the model handles multi-fault scenarios

---

## Author

**Sampan Singh**
Open to opportunities in the SF Bay Area
[GitHub](https://github.com/Ekaum999) • [LinkedIn](https://www.linkedin.com/in/sampan-singh-48365657)
