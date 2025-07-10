
# TML ASSIGNMENT 3: ROBUSTNESS

## Overview

This project implements a robust image classifier using a deep ResNet backbone, designed to perform well on both clean and adversarially perturbed images. The model is trained and evaluated on a 10-class image dataset, with robustness measured against adversarial attacks using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).

## Features

- **ResNet-50 Backbone**: Powerful feature extractor for complex image classification tasks.
- **Data Augmentation**: Includes random crop, horizontal flip, rotation, color jitter, and cutout regularization.
- **Adversarial Training**: Improves robustness against FGSM and PGD attacks.
- **Regularization**: Mixup, label smoothing, gradient clipping, and cosine learning rate scheduling.
- **Flexible Training Pipeline**: Supports clean and adversarial training in separate phases.

## File Structure

```
.
├── robustness.py            # Main script for training, evaluation, and model saving
├── Robustness_Report.pdf    # Project report with methodology and results
├── models/
│   └── robust_model.pt      # Trained model checkpoint (state_dict)
├── train.pt                 # Training dataset (not included publicly)
└── README.md                # This file
```

## How to Use

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10+ and torchvision
- NumPy

Install dependencies with:

```bash
pip install torch torchvision numpy
```

### Training

1. Place the provided `train.pt` dataset file in your project directory.
2. Run the main script:

   ```bash
   python robustness.py
   ```

3. The script will:
   - Train the model first on clean images, then on adversarial images.
   - Save the best-performing model.
   - Print evaluation metrics during training.

### Evaluation

- The script evaluates the model’s accuracy on clean data and under both FGSM and PGD adversarial attacks.
- The final model is saved as `models/robust_model.pt`.

## Key Techniques

- **Adversarial Training**: Enhances robustness by exposing the model to adversarial examples during training.
- **Mixup & Cutout**: Encourage smoother decision boundaries and improved generalization.
- **Label Smoothing**: Reduces overconfidence and helps prevent overfitting.
- **Cosine Annealing LR Scheduler**: Provides smooth learning rate adjustment.
- **Gradient Clipping**: Prevents exploding gradients and stabilizes training.

## Results

| Metric            | Accuracy (%) |
|-------------------|-------------|
| Clean Accuracy    | 68.56       |
| FGSM Robustness   | 13.36       |
| PGD Robustness    | 0.5         |

*For detailed analysis, refer to the project report.*

## Credits

Developed as part of a deep learning course assignment on robustness.


