# 🧠 Melanoma Detection Using Deep Learning (Skin Lesion Classification)

This repository contains a Jupyter notebook that applies deep learning (CNN) to classify dermoscopic skin lesion images and detect melanoma — one of the most dangerous forms of skin cancer.

---

## 🔗 Dataset Used

This project uses the [ISIC Skin Cancer dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic) from the International Skin Imaging Collaboration (ISIC), which contains labeled dermoscopic images of various skin lesions.

The dataset includes 33,000+ RGB images of 7 different classes, helping build a robust skin lesion classifier.

---

## 📁 Contents

* `Melanoma_Detection_using_deep_learning.ipynb`: The complete notebook with preprocessing, CNN architecture, training, and evaluation.
* Visualizations for accuracy, loss, and confusion matrix.

---

## 🎯 Objective

To detect and classify types of skin lesions — especially melanoma — using a Convolutional Neural Network (CNN), improving early diagnosis accuracy through image-based deep learning.

---

## 🛠️ Libraries Used

The following Python libraries are used in this notebook:

* `tensorflow` & `keras` — for deep learning model building  
* `pandas` & `numpy` — for data manipulation  
* `matplotlib` & `seaborn` — for plotting and visualization  
* `sklearn` — for evaluation metrics like accuracy, precision, recall, F1-score  
* `cv2` — for image resizing and processing (optional)

---

## 🔍 Key Steps

1. **Data Loading & Normalization**: Images are resized and pixel values are normalized.
2. **Label Encoding**: Skin lesion categories are encoded numerically.
3. **Train-Test Split**: Dataset is split into training and test sets with stratified sampling.
4. **Model Building**: A CNN model is created using Keras Sequential API.
5. **Training**: The model is trained with appropriate callbacks.
6. **Evaluation**: Metrics like accuracy, precision, recall, and F1-score are computed.
7. **Visualization**: Plots include training/validation accuracy, loss, and a confusion matrix.

---





## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/melanoma-detection-cnn.git
   cd melanoma-detection-cnn
