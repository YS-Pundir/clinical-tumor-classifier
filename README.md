# 🏥 Clinical Tumor Classifier

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Live-Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/yuvrajpundir/clinical-tumor-classifier)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![IIT Roorkee](https://img.shields.io/badge/IIT%20Roorkee-Module%202-blue?style=for-the-badge)](https://www.iitr.ac.in/)
[![UE Potsdam](https://img.shields.io/badge/UE%20Potsdam-Software%20Engineering-red?style=for-the-badge)](https://www.ue-germany.com/)

---

## 📌 Project Overview

This project is a **Diagnostic Inference Engine** developed as part of the **IIT Roorkee: Agentic AI Systems & Design** program (Module 2), while pursuing a **B.Sc. in Software Engineering** at **University of Europe for Applied Sciences (UE), Potsdam**.

The application uses **Machine Learning** to classify breast cancer tumors as:

- **Malignant** → Cancerous
- **Benign** → Non-cancerous / Safe

The primary objective of this project was to master the **Full Machine Learning Lifecycle** — from raw data analysis and feature engineering to model deployment on the cloud using an interactive web application.

---

## 🚀 Live Deployment

The model is officially deployed and accessible globally:

👉 **[Interactive Web Interface on Hugging Face](https://huggingface.co/spaces/yuvrajpundir/clinical-tumor-classifier)**

---

## 🛠️ ML Engineering Workflow

### 1️⃣ Data Analysis & Feature Engineering

- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Features:** 30 clinical measurements including:
  - Mean Radius
  - Texture
  - Perimeter
  - Area
  - Smoothness
- **Scaling Technique:** `StandardScaler`

Feature scaling was implemented to ensure that all numerical inputs contribute equally to the model's learning process and decision-making.

---

### 2️⃣ Model Selection — Why Logistic Regression?

Three different machine learning models were evaluated:

#### ✅ Logistic Regression *(Selected Model)*

- Achieved **98% accuracy** on unseen test data
- Minimal overfitting
- Highly interpretable for medical use-cases
- Computationally efficient

#### 🌳 Decision Tree

- Strong training accuracy
- Tended to overfit at deeper tree depths

#### 🌲 Random Forest

- Excellent performance
- More complex than required for this diagnostic task

After evaluation, **Logistic Regression** was selected because it provided the best balance between:

- Accuracy
- Simplicity
- Interpretability
- Generalization

---

### 3️⃣ Production Pipeline

To prevent **Data Leakage**, a complete Scikit-Learn `Pipeline` was created that combines:

- Data Scaling
- Model Inference

This ensures every input from the Gradio web interface is processed exactly the same way as the training data.

```python
# The "Brain" of the Application

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

---

## 📂 Repository Structure

```text
clinical-tumor-classifier/
│
├── Exports/
│   └── logistic_regression_pipeline.joblib
│
├── notebooks/
│   └── classification_analysis.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

### Folder Explanation

| Folder/File | Purpose |
|---|---|
| `Exports/` | Stores the trained ML pipeline |
| `notebooks/` | Research and experimentation notebook |
| `app.py` | Gradio web application |
| `requirements.txt` | Dependency list for deployment |
| `README.md` | Project documentation |

---

## 🔧 Technical Stack

### Languages
- Python 3.13

### ML & Data Science
- Scikit-Learn
- Pandas
- NumPy

### Model Persistence
- Joblib

### UI & Deployment
- Gradio 6.0
- Hugging Face Spaces

---

## 🎓 IIT Roorkee Journey

This project marks the completion of:

### **Module 2 — Machine Learning Foundations**

under the **IIT Roorkee: Agentic AI Systems & Design Program**.

The next phase of the journey focuses on:

- Generative AI
- RAG Systems
- AI Agents
- Autonomous Workflows

as part of **Module 3: GenAI & Agents**.

---

## 👨‍💻 Developed By

### **Yuvraj Singh Pundir**

- 🎓 Software Engineering Student  
  **University of Europe for Applied Sciences (UE), Potsdam**

- 🤖 IIT Roorkee Agentic AI Fellow

---


