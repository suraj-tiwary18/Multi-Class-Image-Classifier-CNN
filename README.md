# 🖼️ Multi-Class Image Classifier using CNN

A Deep Learning project built using **TensorFlow** and **Keras** that classifies images into **7 different classes** using a Convolutional Neural Network (CNN).

---

## 📌 Features

- Multi-class image classification
- CNN model built from scratch
- Image preprocessing using ImageDataGenerator
- Trained model saved in `.keras` format
- Predicts the class of a new image

---

## 📂 Project Structure

```text
Multi-Class-Image-Classifier-CNN/
│
├── dataset/
│   ├── train/
│   └── test/
│
├── dl_env/                  # Virtual Environment (Not required for GitHub)
├── Model.py                 # Model training
├── Code_of_Run_Model.py     # Predict on new images
├── mult_object_classifier.keras
├── requirements.txt
├── Cat03.jpg
├── images.jpeg
├── human_image.webp
├── photo-1566669086984-077347c1f4bb.avif
├── PATOYS-Batman-Licensed-Electric-Ride-On-Bike-for-Kids...
└── .gitignore
```

---

## 🧠 Classes

The model classifies images into the following categories:

- 🚲 Bike
- 🚗 Car
- 🐱 Cat
- 🐶 Dog
- 🌸 Flowers
- 🐴 Horses
- 👤 Human

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy

---

## ⚙️ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

```bash
python Model.py
```

---

## 🔍 Predict an Image

```bash
python Code_of_Run_Model.py
```

---

## 📈 Model Architecture

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Conv2D (128 Filters)
- MaxPooling2D
- Conv2D (256 Filters)
- MaxPooling2D
- Flatten
- Dense (256)
- Dense (7 - Softmax)

---

## 📦 Output Classes

The trained model predicts one of the following classes:

- Bike
- Car
- Cat
- Dog
- Flowers
- Horses
- Human

---

## 👨‍💻 Author

**Suraj Tiwari**

B.Tech Student | Machine Learning & Deep Learning Learner

GitHub: https://github.com/suraj-tiwary18