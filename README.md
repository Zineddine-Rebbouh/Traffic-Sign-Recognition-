# German Traffic Sign Recognition with CNN

This repository contains a Jupyter Notebook implementing a Convolutional Neural Network (CNN) for classifying German traffic signs from the GTSRB dataset. The project demonstrates end-to-end machine learning workflow, including data preprocessing, augmentation, model building, training, and evaluation.

## Overview
Traffic sign recognition is a critical component of autonomous driving systems and advanced driver-assistance systems (ADAS). This notebook uses a simple CNN architecture to achieve high accuracy on the GTSRB dataset, which consists of 43 classes of traffic signs under varying conditions like lighting, occlusion, and weather.

Key features:
- Data loading and exploration with visualizations (e.g., class distribution).
- Image preprocessing and normalization.
- Data augmentation to handle class imbalance and improve generalization.
- CNN model with convolutional layers, pooling, dropout, and dense layers.
- Training with Adam optimizer and categorical cross-entropy loss.
- Evaluation using accuracy, loss, classification reports, and confusion matrices.

## Dataset
The notebook uses the [GTSRB dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) available on Kaggle:
- Training samples: ~39,209 images.
- Test samples: ~12,630 images.
- Classes: 43 (e.g., speed limits, yield signs, no entry).
- Image size: Resized to 32x32 pixels for efficiency.

The dataset includes CSV metadata files (`Train.csv` and `Test.csv`) with image paths and labels.

## Requirements
To run the notebook, you'll need:
- Python 3.11+
- Libraries (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - opencv-python (cv2)
  - scikit-learn
  - tensorflow (or keras)
  - Optional: Jupyter Notebook or Kaggle environment

Example `requirements.txt`:
