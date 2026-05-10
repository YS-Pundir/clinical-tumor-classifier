# 🏥 Clinical Tumor Classifier 

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Live-Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/yuvrajpundir/clinical-tumor-classifier)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![IIT Roorkee](https://img.shields.io/badge/IIT%20Roorkee-Module%202-blue?style=for-the-badge)](https://www.iitr.ac.in/)

## 📌 Project Overview
This project is a **Diagnostic Inference Engine** developed as part of the **IIT Roorkee: Agentic AI Systems & Design** program (Module 2). It utilizes Machine Learning to classify breast cancer tumors as **Malignant** (Cancerous) or **Benign** (Safe) based on cellular measurements.

The objective was to master the **Full ML Lifecycle**: moving from raw data exploration to a cloud-deployed application that can be used by healthcare professionals.

## 🚀 Live Deployment
The model is officially deployed and accessible globally:
👉 **[Interactive Web Interface on Hugging Face](https://huggingface.co/spaces/yuvrajpundir/clinical-tumor-classifier)**

---

## 🛠️ The ML Engineering Workflow

### 1. Data Analysis & Feature Engineering
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset.
- **Features:** 30 clinical parameters including Mean Radius, Texture, Perimeter, and Smoothness.
- **Scaling:** Used `StandardScaler` to ensure all features contribute equally to the model's decision-making process.

### 2. Model Selection (Why Logistic Regression?)
I evaluated three different architectures to find the best balance between training performance and generalization:
- **Logistic Regression:** Selected as the winner. It achieved **98% accuracy** on test data with minimal overfitting.
- **Decision Trees:** Showed a tendency to memorize training data (Overfitting) at infinite depth.
- **Random Forest:** Strong results, but Logistic Regression was chosen for its mathematical simplicity and high interpretability in medical contexts.

### 3. Production Pipelines
To prevent **Data Leakage**, I built a Scikit-Learn `Pipeline` that bundles the Scaler and the Estimator together. This ensures that any input from the Gradio UI is pre-processed exactly like the training data before prediction.

```python
# The "Brain" of the application
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

📂 Repository Structure

clinical-tumor-classifier/
├── Exports/
│   └── logistic_regression_pipeline.joblib  # The "Frozen" Model & Scaler
├── notebooks/
│   └── classification_analysis.ipynb        # Data Science Research
├── app.py                                   # Gradio Web Interface Script
├── requirements.txt                         # Cloud Dependency List
└── README.md                                # Project Documentation

🔧 Technical Stack
Languages: Python 3.13

ML Frameworks: Scikit-Learn, Pandas, NumPy

Persistence: Joblib

UI/Deployment: Gradio 6.0, Hugging Face Spaces

🎓 IIT Roorkee Journey
This project marks the completion of Module 2: Machine Learning Foundations . I am now moving into Module 3: GenAI & Agents , where I will apply these foundations to build RAG systems and autonomous AI Agents.

👨‍💻 Developed By
Yuvraj Singh Pundir  Software Engineering Student @ University of Europe, Potsdam  IIT Roorkee Agentic AI Fellow
