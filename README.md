
# Pneumonia Detection-XAI 🫁

A deep learning web application for **pneumonia detection from chest X-rays** using an ensemble of CNN models with explainable AI (XAI) heatmap visualization.

## 🔍 Overview

Pneumonia Detection-XAI combines **VGG**, **ResNet**, and **DenseNet** models in an ensemble to classify chest X-rays as **NORMAL** or **PNEUMONIA**, and generates a fused activation heatmap to visually explain the model's decision.

## ✨ Features

- Ensemble prediction using VGG, ResNet, and DenseNet
- PneumoFusion-XAI heatmap overlay for explainability
- Pneumonia severity estimation (Mild / Moderate / Severe)
- X-ray image validation (rejects non-X-ray uploads)
- Downloadable PDF diagnostic report
- Drag & drop web interface built with Flask

## 🛠️ Tech Stack

- **Backend:** Python, Flask, TensorFlow/Keras, OpenCV
- **Frontend:** HTML, CSS, JavaScript
- **PDF Generation:** fpdf
- **Models:** VGG16, ResNet50, DenseNet121 (fine-tuned)

## 🚀 Getting Started

### Prerequisites

```bash
pip install flask tensorflow opencv-python fpdf numpy
```

### Run the App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## 📁 Project Structure

```
epicsproject/
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── uploads/            # Uploaded X-rays
│   └── results/            # Generated heatmaps
├── vgg_finetuned.h5
├── resnet_finetuned.h5
├── densenet_model_trained.h5
└── dataset/
    ├── train/
    ├── val/
    └── test/
```

## 📊 Dataset

Uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle with `NORMAL` and `PNEUMONIA` classes.

> **Note:** The `dataset/` folder is not included in this repository due to size. Download it from the Kaggle link above and place it in the project root as shown in the project structure.

## 🤖 Pretrained Models

The `.h5` model files are not included in this repository due to GitHub's file size limits. Download them from the link below and place them in the project root:

📥 **[Download Models – Google Drive](https://drive.google.com/drive/folders/1cFSpuEhBn576aHNkiKgfnhKuwPhUgMCb?usp=sharing)**

Required files:
- `vgg_finetuned.h5`
- `resnet_finetuned.h5`
- `densenet_model_trained.h5`

## 📄 Output

- Prediction label with confidence score
- Severity level with intensity percentage
- PneumoFusion-XAI heatmap overlay
- Downloadable PDF diagnostic report

## ⚠️ Disclaimer

This tool is for **research and educational purposes only** and is not a substitute for professional medical diagnosis.
=======
