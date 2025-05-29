
# 🌿 Leaf Disease Detection Web App

A Flask-based web application that uses a deep learning model to detect plant leaf diseases from images and suggests appropriate treatments. Built with Python, PyTorch, and ResNet-18, the app is optimized for cloud deployment (Heroku/AWS) and features a responsive frontend with a clean dark-green theme.

## 📌 Table of Contents

- [Introduction](#-introduction)
- [Objective](#-objective)
- [Tools and Technologies](#-tools-and-technologies)
- [Methodology](#-methodology)
- [Features](#-features)
- [Results](#-results)
- [Conclusion](#-conclusion)
- [Future Enhancements](#-future-enhancements)
- [References](#-references)
- [How to Run Locally](#-how-to-run-locally)

## 🌱 Introduction

Plant leaf diseases are a major threat to global food security. This application provides early detection and diagnosis using machine learning. Users can upload an image of a leaf and receive real-time predictions with treatment recommendations.

## 🎯 Objective

To develop a user-friendly web app that:
- Classifies leaf diseases from uploaded images using a trained CNN.
- Provides actionable treatment suggestions.
- Is ready for deployment on cloud platforms like Heroku or AWS.

## 🛠️ Tools and Technologies

- **Python** – Core language  
- **PyTorch** – For building and training the model  
- **ResNet-18** – Transfer learning for image classification  
- **Flask** – Web framework for backend  
- **HTML5, CSS3, JavaScript** – For frontend UI  
- **Heroku / AWS** – Deployment platforms  

## 🔍 Methodology

### 1. Dataset Preparation
- Used an augmented dataset of healthy and diseased leaf images.
- Each class was organized in separate folders for easy labeling.

### 2. Preprocessing
- Resized all images to 224x224 pixels.
- Normalized using ImageNet statistics.
- Split into 80/20 training and testing sets.

### 3. Model Training
- Utilized ResNet-18 with pretrained weights.
- Final layers retrained using CrossEntropyLoss and Adam optimizer.
- Metrics monitored to avoid overfitting.

### 4. Flask Integration
- Backend handles image upload, prediction, and treatment lookup.
- Frontend styled with a dark-green themed UI for ease and relevance.

### 5. Deployment Preparation
- Configured `requirements.txt`, `Procfile`, and `runtime.txt` for Heroku/AWS.

## 🌟 Features

- ✅ Accurate disease classification with confidence scores  
- ✅ Real-time treatment recommendations  
- ✅ Upload and predict directly via web interface  
- ✅ Clean and responsive dark-green UI  
- ✅ Ready for cloud deployment  

## 📈 Results

- High accuracy on test dataset  
- Fast and reliable performance  
- Intuitive frontend with helpful outputs (disease name, confidence, treatment)  
- Easily extendable and maintainable codebase  

## ✅ Conclusion

This project demonstrates how deep learning and web technologies can be combined to deliver smart agricultural tools. It's scalable, practical, and user-friendly—ready to support real-world crop management needs.

## 🔮 Future Enhancements

- 📷 Add more plant species and disease categories  
- 🌍 Multilingual and voice-based input support  
- 📑 Batch image upload and downloadable reports  
- 🧠 Feedback loop to enhance recommendations  
- 📱 Mobile-friendly PWA version  

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)  
- [Flask Documentation](https://flask.palletsprojects.com/)  
- [Augmented Plant Leaf Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)  

## 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/leaf-disease-detection-app.git
   cd leaf-disease-detection-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser and go to:
   ```
   http://localhost:5000
   ```

> Developed with ❤️ for smart farming.
