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
```
pandas
numpy
matplotlib
seaborn
opencv-python
scikit-learn
tensorflow
```

No GPU is required, but it speeds up training (the notebook is set for CPU acceleration on Kaggle).

## Usage
1. **Clone the repository**:
   ```
   git clone https://github.com/your-username/german-traffic-sign-recognition.git
   cd german-traffic-sign-recognition
   ```

2. **Download the dataset**:
   - Place the GTSRB dataset in a folder named `gtsrb-german-traffic-sign` (or update `base_folder` in the notebook).
   - Ensure `Train.csv` and `Test.csv` are present.

3. **Run the notebook**:
   - Open `traffic-sign-recognition.ipynb` in Jupyter or upload to Kaggle.
   - Execute cells sequentially. Training takes ~10-20 minutes for 10 epochs on CPU.
   - Adjust hyperparameters like `img_size`, `epochs`, or augmentation settings as needed.

4. **Train and Evaluate**:
   - The model is trained with data augmentation (rotation, zoom, shifts).
   - Results include:
     - Training/validation accuracy and loss plots.
     - Test accuracy (typically >95% with the given setup).
     - Classification report (precision, recall, F1-score per class).
     - Confusion matrix heatmap.

## Model Architecture
A simple Sequential CNN:
- Conv2D (32 filters) → MaxPooling2D
- Conv2D (64 filters) → MaxPooling2D
- Flatten → Dense (256 units, ReLU) → Dropout (0.5)
- Dense (43 units, softmax)

Compiled with Adam optimizer and categorical cross-entropy. Input shape: (32, 32, 3).

## Results
- **Test Accuracy**: ~96-98% (varies by run; improve with more epochs or deeper model).
- **Challenges**: Class imbalance (visualized in the notebook) may affect minority classes.
- **Visualizations**: Class distribution bar plot, training history plots, confusion matrix.
- Example output: High precision/recall for common signs (e.g., speed limits), potential confusion between similar signs (e.g., yield vs. stop).

For detailed results, run the notebook and review the classification report and confusion matrix.

## Improvements
- Add more layers (e.g., deeper CNN like ResNet) for better accuracy.
- Use transfer learning (e.g., MobileNetV2).
- Handle class imbalance with weighted loss or oversampling.
- Deploy as a web app using TensorFlow.js or Flask.

## License
MIT License. Feel free to use, modify, and distribute.

## Acknowledgments
- Dataset: GTSRB from Kaggle.
- Built with TensorFlow/Keras.

If you find this useful, star the repo or open an issue for suggestions!
