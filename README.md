# COVID-19 Image Classification

This repository provides a complete pipeline for image classification using TensorFlow and Keras. The project utilizes the VGG16 model as a base for transfer learning to classify images into two categories. It includes data preprocessing, model training, evaluation, and visualization.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates a deep learning approach for classifying images related to COVID-19. The model is based on VGG16, a popular convolutional neural network architecture, fine-tuned with additional custom layers to adapt to specific classification tasks.

## Installation

To set up and run this project, follow these steps:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/covid19-image-classification.git
    cd covid19-image-classification
    ```

2. **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

   Ensure `requirements.txt` includes all necessary libraries. For this project, you will need:
   - `tensorflow`
   - `opencv-python`
   - `scikit-learn`
   - `matplotlib`
   - `imutils`
   - `geopy`

## Usage

1. **Prepare the Dataset:**
   - Place your image dataset in a directory structure where each subdirectory represents a class label.
   - Update the `--dataset` argument to point to this directory.

2. **Run the Script:**
    ```bash
    python script.py --dataset path/to/dataset --plot path/to/output/plot.png --model path/to/output/model.h5
    ```
   - `--dataset`: Path to the input dataset.
   - `--plot`: Path where the training/validation loss and accuracy plot will be saved.
   - `--model`: Path where the trained model will be saved.

## Features

- **Data Preprocessing:** Loads and preprocesses images, including resizing and normalization.
- **Model Architecture:** Utilizes VGG16 as a base model with additional layers for custom classification.
- **Training:** Implements data augmentation and trains the model with specified hyperparameters.
- **Evaluation:** Provides metrics such as accuracy, sensitivity, and specificity, and saves evaluation plots.
- **Visualization:** Plots training loss and accuracy over epochs, and saves the plot to a file.

## Results

- **Model Evaluation:**
  - Classification report including precision, recall, and F1-score.
  - Confusion matrix with computed accuracy, sensitivity, and specificity.

- **Training/Validation Plot:**
  - Plot of training loss and accuracy, and validation loss and accuracy over epochs.
  - Saved as an image file specified by `--plot`.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please fork this repository, make your changes, and submit a pull request. Ensure your code adheres to the existing style and includes relevant documentation.
