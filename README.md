# Fashion MNIST KNN Classifier

This project implements a **K-Nearest Neighbors (KNN)** classifier on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
It includes functions for downloading the dataset, preprocessing data, visualizing examples, tuning hyperparameters, and evaluating classification performance with and without **Principal Component Analysis (PCA)** for dimensionality reduction.

---

## Overview

The Fashion MNIST dataset consists of 70,000 grayscale images of 10 clothing categories such as shirts, shoes, and bags.  
Each image is **28x28 pixels**, representing one of the following classes:

| Label | Class Name   |
|:------|:--------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

This code builds a **KNN classifier** to recognize these images and evaluates performance through accuracy metrics, per-class accuracy, and confusion matrices.

---

## ⚙️ Features

- **Automatic dataset download** from official URLs.  
- **Gzip decompression** for image and label files.  
- **Visualization** of random samples from the dataset.  
- **KNN classification** with configurable hyperparameters.  
- **Accuracy computation** (overall, per-class, and confusion matrix).  
- **Hyperparameter tuning** for optimal `k` and weighting strategy.  
- **Optional PCA** for dimensionality reduction and improved performance.  
- **Clear printed outputs** for model accuracy and confusion matrix visualization.

---

## 🧠 Main Functions

### `download_fashion_mnist(url, file_name)`
Downloads a dataset file from the specified URL if it does not already exist.

### `load_fashion_mnist(image_file, label_file)`
Loads image and label data from compressed `.gz` files and reshapes them into usable arrays.

### `show_random_images(images, labels, class_names, num_images=9)`
Displays a random grid of Fashion MNIST images with class labels using Matplotlib.

### `classify_images(image_labels, class_names)`
Counts how many images exist per class and prints a summary.

### `compute_accuracy(test_pairs, num_classes)`
Computes overall accuracy of predictions vs actual labels.

### `compute_per_class_accuracy(test_pairs, num_classes)`
Computes accuracy for each individual class.

### `compute_confusion_matrix(test_pairs, num_classes)`
Generates a confusion matrix showing correct and incorrect classifications.

### `tune_classifier(train_vectors, train_labels)`
Performs hyperparameter tuning for KNN using a validation split.  
Searches over multiple values of `k` and weighting strategies (`uniform` and `distance`).

### `tune_classifier_with_pca(train_vectors, train_labels, test_vectors, test_labels)`
Performs PCA to reduce data dimensionality, then tunes and evaluates the KNN classifier for improved performance and efficiency.

---

## 🧾 Example Workflow

1. **Download and load** the Fashion MNIST dataset (both training and test sets).  
2. **Visualize** sample images using `show_random_images()`.  
3. **Normalize** and flatten the image data for model input.  
4. **Train** a basic KNN classifier.  
5. **Evaluate** using accuracy and confusion matrix metrics.  
6. **Tune hyperparameters** for improved accuracy.  
7. **Run PCA-based optimization** for dimensionality reduction and faster computation.

---

## 📊 Metrics and Evaluation

The script computes the following:

- **Overall Accuracy**:  
  Fraction of correctly predicted labels over the total.

- **Per-Class Accuracy**:  
  Accuracy calculated individually for each of the 10 clothing classes.

- **Confusion Matrix**:  
  Displays classification performance across all class combinations.

Example (formatted output):

