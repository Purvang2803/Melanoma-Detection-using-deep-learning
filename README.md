# 🧠 Melanoma Detection Using Deep Learning (Skin Lesion Classification)

This repository contains a Jupyter notebook that applies deep learning (CNN) to classify dermoscopic skin lesion images and detect melanoma — one of the most dangerous forms of skin cancer.

---

## 🔗 Dataset Used

This project uses the [HAM10000 ("Human Against Machine with 10000 training images") dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), which includes labeled dermoscopic images of different types of skin lesions.

The dataset contains over 10,000 high-resolution RGB images divided into the following 7 classes:

- **akiec**: Actinic keratoses and intraepithelial carcinoma
- **bcc**: Basal cell carcinoma
- **bkl**: Benign keratosis-like lesions
- **df**: Dermatofibroma
- **nv**: Melanocytic nevi
- **vasc**: Vascular lesions
- **mel**: Melanoma

---

## 📁 Contents

* `melanoma_classification_cnn.ipynb`: The complete notebook with preprocessing, CNN model, training, and evaluation.
* Data visualizations: count plots, training/validation curves, and confusion matrix.
* Evaluation metrics: accuracy, precision, recall, and F1-score.

---

## 🎯 Objective

To accurately detect and classify skin lesions — with a primary focus on melanoma — using a Convolutional Neural Network (CNN), aiding in early detection and reducing misdiagnosis risks.

---

## 🛠️ Libraries Used

The notebook uses the following Python libraries:

- `tensorflow`, `keras`: For deep learning model development  
- `numpy`, `pandas`: For data handling and analysis  
- `matplotlib`, `seaborn`: For visualizations  
- `sklearn`: For model evaluation and performance metrics  
- `cv2`: For image processing (resizing, color conversion)

---

## 🔍 Key Steps

1. **Data Preprocessing**
   - Load metadata and image files from directory
   - Normalize and resize images
   - Handle imbalanced data using class weights or augmentation

2. **Exploratory Data Analysis**
   - Countplot of lesion types
   - Sample image grid from each class

3. **Label Encoding**
   - Encode categorical labels (`dx`) into numerical values

4. **Train-Test Split**
   - Stratified sampling to maintain class distribution
   - Separate training, validation, and test sets

5. **Model Building**
   - Custom CNN and EfficientNetB0 transfer learning
   - Dropout, BatchNormalization, ReLU activation

6. **Training**
   - Use of early stopping and learning rate reduction
   - Epoch-wise performance tracking

8. **Results Visualization**
   - Accuracy/Loss curves
   - Model predictions vs true labels

---

## 📊 Visuals Included

- 📈 **Training vs Validation Accuracy/Loss**
- 📊 **Confusion Matrix Heatmap**
- 🔍 **Sample Image Predictions**

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/melanoma-ham10000-cnn.git
   cd melanoma-ham10000-cnn
