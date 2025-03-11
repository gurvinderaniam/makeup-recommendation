# 🎨 AI-Based Makeup Recommendation System  

## 📌 Project Overview  
This project is an **AI-powered system** that analyzes **skin tone & face shape** to recommend the best makeup products. The system extracts facial features from images using **computer vision & machine learning** and provides personalized makeup suggestions.  

---

## 🔹 Features  
✅ **Facial Feature Extraction** (Skin Tone & Face Shape)  
✅ **Makeup Recommendation Based on AI Model**  
✅ **Flask API for Model Deployment**  
✅ **Streamlit Web UI for Easy Interaction**  
✅ **Trained Machine Learning Model**  

---

## 🔹 Tech Stack  
- **Python (OpenCV, Dlib, NumPy, Pandas)**  
- **Machine Learning (Scikit-Learn, Random Forest)**  
- **Deep Learning (TensorFlow/Keras for Feature Extraction)**  
- **Flask (API Deployment)**  
- **Streamlit (Web UI)**  
- **Git & GitHub for Version Control**  

---

## 🔹 Installation Guide 

pip install -r requirements.txt
3️⃣ Run Flask API

python flask_api.py
📌 The API will be available at: http://127.0.0.1:5000/predict

4️⃣ Run Streamlit Web App

streamlit run app_ui.py
📌 Access the Web UI at http://localhost:8501

🔹 How It Works
1️⃣ Upload an Image via Web UI
2️⃣ AI extracts skin tone & face shape
3️⃣ Machine Learning model predicts the best makeup
4️⃣ User gets a personalized makeup recommendation

🔹 Dataset
4220+ images processed for skin tone & face shape
Extracted RGB values & shape classification
Dataset stored in:
skin_tone_data.csv
face_shape_data_fixed.csv
final_dataset.csv
🔹 Future Enhancements
🔹 Deploy on Cloud (AWS/GCP) for public access
🔹 Improve Model with Deep Learning (CNNs)
🔹 Expand Dataset for Better Generalization
🔹 User Feedback Loop for Model Improvement

🔹 Contributing
🚀 Contributions are welcome! Feel free to fork the repo and submit a pull request.
