# MyanNet — Lightweight CNN for Burmese Handwritten Digit Recognition

> **99.49% test accuracy · 10,634 trainable parameters · 24.18 KB TFLite model · 0.263 ms inference**

MyanNet is a lightweight convolutional neural network for classifying handwritten Burmese digits (၀–၉) from the [BHDD dataset](https://arxiv.org/abs/2603.21966). It combines depthwise separable convolutions, global average pooling, and batch normalization to achieve near state-of-the-art accuracy at a fraction of the parameter count — making it deployable on edge devices such as Android phones and Raspberry Pi.
---

## Table of Contents

- [Results](#results)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Reproducing Results](#reproducing-results)
- [TFLite Export](#tflite-export)
- [Citation](#citation)

---

## Results

### Model Comparison (BHDD Test Set)

| Model | Trainable Params | Test Accuracy |
|---|---|---|
| Baseline CNN | 34,826 | 99.58% |
| GAP-BN CNN | 21,418 | 99.51% |
| **MyanNet (proposed)** | **10,634** | **99.49%** |
| CNN (myMNIST benchmark) | 34,826 | 99.70% |
| Improved CNN w/ BN+Aug (BHDD paper) | ~431K | 99.83% |

MyanNet achieves a **69.5% parameter reduction** over the Baseline CNN at a cost of only 0.09 percentage points of accuracy.

### 5-Fold Cross-Validation

| Fold | Test Accuracy |
|---|---|
| Fold 1 | 99.40% |
| Fold 2 | 99.38% |
| Fold 3 | 99.55% |
| Fold 4 | 99.52% |
| Fold 5 | 99.44% |
| **Mean ± Std** | **99.46% ± 0.06%** |

### TFLite Deployment (1,000 inference runs, CPU)

| Metric | Value |
|---|---|
| Model size | 24.18 KB |
| Mean latency | 0.263 ms / image |
| P50 latency | 0.262 ms / image |
| P95 latency | 0.299 ms / image |

---

## Architecture

MyanNet uses a two-block convolutional design followed by a lightweight classification head:

```
Input (28×28×1)
│
├── Block 1 — Standard Convolution
│   ├── Conv2D (3×3, 64 filters, same padding, ReLU)
│   ├── BatchNormalization
│   └── MaxPooling2D (2×2)  →  14×14×64
│
├── Block 2 — Depthwise Separable Convolution
│   ├── DepthwiseConv2D (3×3, same padding, ReLU)
│   ├── BatchNormalization
│   ├── Conv2D (1×1, 64 filters)   ← pointwise mixing
│   ├── BatchNormalization
│   └── MaxPooling2D (2×2)  →  7×7×64
│
└── Head
    ├── GlobalAveragePooling2D  →  64-dim vector
    ├── Dropout (0.19)
    ├── Dense (64, ReLU)
    └── Dense (10, Softmax)
```

**Total trainable parameters: 10,634**

### Hyperparameters (found via Optuna, 30 trials)

| Hyperparameter | Value |
|---|---|
| filters1 | 64 |
| filters2 | 64 |
| dropout | 0.194 |
| learning_rate | 3.00 × 10⁻³ |
| dense_units | 64 |

---

## Dataset

MyanNet is trained and evaluated on the **Burmese Handwritten Digit Dataset (BHDD)**:

- **87,561** grayscale images (28×28 pixels), 10 classes (digits ၀–၉)
- **60,000** training samples (perfectly balanced, 6,000 per class)
- **27,561** test samples (naturally imbalanced: 389–6,856 per class)
- Format: identical to MNIST (pickle file)

Download from BHDD dataset from Github: https://github.com/baseresearch/BHDD

After downloading, place `data.pkl` at:
```
/kaggle/input/datasets/ahmaungoo/bhdd-set/data.pkl
```
or update the `DATA_PATH` variable in the notebook.

---

## Project Structure

```
MyanNet/
├── MyanNet.ipynb          # Main notebook (training, evaluation, export)
├── README.md
├── requirements.txt
└── outputs/               # Generated after running the notebook
    ├── myannet_best.keras
    ├── myannet_quantized.tflite
    ├── best_params.json
    ├── kfold_results.json
    ├── benchmark_results.json
    ├── results_summary.txt
    ├── confusion_matrix.png
    ├── training_curves.png
    ├── model_comparison.png
    ├── kfold_results.png
    ├── optuna_results.png
    ├── misclassified_samples.png
    ├── sample_images.png
    ├── class_distribution.png
    └── augmentation_preview.png
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- A BHDD data file (`data.pkl`)

### Installation

```bash
git clone https://github.com/<your-username>/MyanNet.git
cd MyanNet
pip install -r requirements.txt
```

### Running the Notebook

Open `MyanNet.ipynb` in Jupyter or run on [Kaggle](https://www.kaggle.com/) (recommended for GPU access).

```bash
jupyter notebook MyanNet.ipynb
```

All sections run top-to-bottom. Outputs are saved to the working directory.

---

## Reproducing Results

All experiments use a fixed seed of `42` set across Python, NumPy, and TensorFlow before any imports. The full pipeline is:

1. **Section 1** — Environment setup and seed fixing
2. **Section 2** — Data loading and preprocessing
3. **Section 3** — Data augmentation configuration
4. **Section 4** — Model definitions (Baseline, GAP-BN, MyanNet)
5. **Section 5** — Optuna hyperparameter search (30 trials)
6. **Section 6** — Final MyanNet training (up to 100 epochs)
7. **Section 7** — 5-fold stratified cross-validation
8. **Section 8** — Evaluation and classification report
9. **Section 9** — Confusion matrix and training curves
10. **Section 10** — Model comparison
11. **Section 11** — TFLite export and inference benchmarking
12. **Section 12** — Results summary

---

## TFLite Export

The notebook exports a post-training integer-quantized TFLite model:

```python
# Load and run inference with the quantized model
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="myannet_quantized.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = np.expand_dims(your_28x28_image / 255.0, axis=(0, -1)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()

predicted_class = np.argmax(interpreter.get_tensor(output_details[0]['index']))
```

---

## Acknowledgements

- BHDD dataset by Swan Htet Aung et al.
- myMNIST benchmark by Ye Kyaw Thu et al.
- MobileNets (Howard et al.) and Network In Network (Lin et al.) for architectural inspiration
- [Optuna](https://optuna.org/) for hyperparameter optimization
