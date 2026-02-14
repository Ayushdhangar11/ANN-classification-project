


# 🚀 Customer Churn Prediction using Artificial Neural Network (ANN)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

This project predicts whether a bank customer is likely to **churn (leave the bank)** using a **Deep Learning model (ANN)** built with TensorFlow/Keras and deployed using Streamlit.

It helps businesses proactively identify high-risk customers and reduce revenue loss.

---

## 🧠 Business Problem

Customer churn directly impacts revenue.

By predicting churn probability:

* Banks can offer retention incentives
* Improve customer engagement
* Reduce financial losses
* Make data-driven strategic decisions

---

## 🏗️ Model Architecture

Artificial Neural Network (ANN):

```python
Dense(units=6, activation='relu')
Dense(units=6, activation='relu')
Dense(units=1, activation='sigmoid')
```

* Hidden Layers → ReLU
* Output Layer → Sigmoid (Binary Classification)
* Loss → Binary Crossentropy
* Optimizer → Adam

---

## 📊 Features Used

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Account Balance
* Number of Products
* Credit Card Ownership
* Active Membership
* Estimated Salary

---

## ⚙️ Data Preprocessing Pipeline

✔ Label Encoding (Gender)
✔ One Hot Encoding (Geography)
✔ Feature Scaling (StandardScaler)
✔ Train-Test Split (80-20)

---

## 📈 Model Performance

> (Update these with your actual results)

* Accuracy: **~85%**
* Loss: ~0.35
* Precision: ~0.82
* Recall: ~0.75

---

## 📂 Project Structure

```
annclassificationProject/
│
├── ann_model.h5
├── scaler.pkl
├── one_hot_encoder_geography.pkl
├── label_encoder_gender.pkl
│
├── experiment.ipynb
├── prediction.ipynb
├── app.py
├── README.md
```

---

## 💻 Streamlit Web Application

The app allows users to:

* Enter customer details
* Predict churn instantly
* View churn probability score

### ▶ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🖥️ Application Preview

(Add screenshots here after uploading to GitHub)

```markdown
![App Screenshot](screenshots/app_preview.png)
```

---

## 🔬 Technical Highlights

* Deep Learning for classification
* Clean feature engineering
* Proper encoding alignment between training & deployment
* Production-safe column ordering
* Streamlit interactive UI
* Modular project structure

---

## 🚀 Future Improvements

* Implement full `Pipeline + ColumnTransformer`
* Hyperparameter tuning with GridSearch / KerasTuner
* SHAP model explainability
* Docker containerization
* Deploy on AWS / Streamlit Cloud
* Add performance dashboard

---

## 📦 Installation

Clone repository:

```bash
git clone https://github.com/yourusername/customer-churn-ann.git
cd customer-churn-ann
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Example Prediction Output

```
Customer is likely to churn ⚠️
Churn Probability: 0.78
```

---

## 📊 Business Impact

This solution enables:

* Early churn detection
* Targeted retention campaigns
* Increased customer lifetime value
* Reduced operational cost

---

## 👨‍💻 Author

**Ayush Dhangar**
Final Year IT Student | Deep Learning & GenAI Enthusiast

---

## 🌟 Why This Project Stands Out

✔ Deep Learning implementation
✔ Real-world business use case
✔ Deployment with interactive UI
✔ Clean modular ML pipeline
✔ Resume-ready production project

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to fork!

---

