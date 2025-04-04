# CIFAKE: Real vs AI-Generated Image Classification

## Hackathon Submission - IIT Bhubaneswar PVH_ML

```markdown

# CIFAKE: Real vs AI-Generated Image Classification

## Hackathon Submission - IIT Bhubaneswar PVH_ML

This repository contains the code, trained models and result files for the ML Hackathon organized at IIT Bhubaneswar. Our task was to build a model capable of classifying whether an image is real or AI-generated using the CIFAKE dataset.

---

---



---

## Model Overview

We used a ResNet50 backbone (pre-trained on ImageNet) for feature extraction. The head consists of global average pooling, batch normalization, dense layers with dropout and a sigmoid output for binary classification.

- Input size: 32x32x3 (resized)
- Base model: `ResNet50` (frozen layers)
- Head:
  - `GlobalAveragePooling2D`
  - `Dense(256, relu)` with L2 regularization
  - `Dropout` (tuned using Keras Tuner)
  - `Dense(64, relu)`
  - `Dense(1, sigmoid)` output

---

## Hyperparameter Tuning

We tuned dropout rate and learning rate using:
- **Keras Tuner RandomSearch** (due to time constraints)
- Search space:
  - Dropout: [0.2, 0.5]
  - Learning rate: [1e-5, 1e-2] (log scale)

The best model was saved as:  
`ResNet50_CIFAKE_best.h5`

---

##  Evaluation Criteria

Tested on **two datasets**:
- Test_dataset_1: Clean images (40% weight)
- Test_dataset_2: Adversarial perturbed images (60% weight)

### Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

---

## Pretrained Model Usage

-  We used `ResNet50` pretrained on ImageNet.
-  All convolutional layers were frozen during training.

---

## Submission Files

-  `Test_1_results.csv` – predictions on clean test set
-  `Test_2_results.csv` – predictions on adversarial test set
-  `ResNet50_CIFAKE_best.h5` – final trained model
-  `README.md`
-   All source code and notebooks

---

## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run notebook:
   `notebooks/train_exploration.ipynb`

---



