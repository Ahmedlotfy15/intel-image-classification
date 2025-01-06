# Intel Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) model using Keras to classify images from the Intel Image Classification dataset. The dataset contains six categories: buildings, forest, glacier, mountain, sea, and street.

## Project Structure

- `train_path`: Path to the training dataset.
- `test_path`: Path to the testing dataset.
- `pred_path`: Path to prediction dataset for unseen images.
- CNN model implemented using Keras.
- Data preprocessing: Resizing images to 100x100.
- Training and evaluating a CNN model.

## Dataset

### Classes:
1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

The dataset is structured as follows:
- `seg_train`: Training images organized by class.
- `seg_test`: Testing images organized by class.
- `seg_pred`: Unlabeled images for prediction.

## Prerequisites

To run this project, install the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn tensorflow keras opencv-python
```

## Code Workflow

### 1. Data Loading and Preprocessing
- Load images from the dataset directories.
- Resize all images to 100x100.
- Convert data to NumPy arrays for model compatibility.

### 2. Visualization
- Display random samples from the training, testing, and prediction datasets.

### 3. CNN Model
- Built using Keras Sequential API.
- Architecture:
  - Multiple Conv2D layers with ReLU activation.
  - MaxPooling layers for dimensionality reduction.
  - Fully connected Dense layers for classification.
  - Dropout layer to reduce overfitting.

### 4. Training
- Compile the model with Adam optimizer and sparse categorical crossentropy loss.
- Train the model for 50 epochs with a batch size of 64.

### 5. Evaluation
- Evaluate the model on the test set.
- Output test loss and accuracy.

### 6. Prediction
- Predict categories for images in the prediction dataset.
- Display predictions alongside the images.


