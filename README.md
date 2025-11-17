<p align="center">
  <img src="assets/banner.png" alt="Spotify Popularity Prediction Banner" width="850">
</p>

# 🎵 Predicting Spotify Track Popularity (2012–2021)

### FAU ISC 4941 – Data Science Capstone Project

<p align="center">
  
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn)
![FAU](https://img.shields.io/badge/FAU-Data%20Science-blue)

</p>

---

## 📖 Overview
This project explores whether we can **predict a song’s popularity on Spotify** based solely on its **audio features and metadata**. Using more than **62,000 tracks released from 2012–2021**, we trained machine learning models to identify which characteristics—such as **energy, danceability, loudness, and valence**—are most strongly associated with high popularity.

Spotify’s popularity score ranges from **0 to 100**, but because the metric is influenced by several complex factors (stream counts, recency, user engagement), we transformed it into a **binary classification**:  
- **Popular** (top 20%)  
- **Not popular** (bottom 80%)  

Our goal is not only prediction but also **understanding** what makes music resonate in today’s streaming landscape.

---

## 🎯 Objective
The main objective of this project is to build a machine learning model capable of estimating a track’s popularity using **audio features only**, while uncovering important insights into how modern musical elements influence listener preferences.

---

## 💡 Key Findings
- After filtering the dataset to include **tracks released after 2012**, we analyzed **62,015 tracks**.
- A **Gradient Boosting Classifier** produced the strongest results with **~75% accuracy**, balanced precision, and strong AUC.
- Top predictors of popularity were:  
  - **Energy**  
  - **Danceability**  
  - **Loudness**  
  - **Release Year**
- Model probability calibration improved interpretability and produced realistic predictions.
- High-energy, modern, danceable tracks tended to perform significantly better in streaming environments.

---

## 🧠 Methodology

### **1. Data Source**
- **Spotify Dataset 1921–2020** from Kaggle  
- Extended reference from Spotify Web API documentation  
- Focused on tracks **2012–2021** (modern scoring era after the introduction of Spotify’s “Follow” feature)

### **2. Data Preparation**
- Filtered out pre-2012 tracks  
- Cleaned missing values  
- Standardized date formats  
- Created binary label `is_popular`  
- Normalized numerical features when necessary  
- Train/test split (80/20 stratified)

### **3. Modeling**
Models tested:
- **Linear Regression** (baseline)
- **Random Forest**
- **Gradient Boosting Classifier** (best performance)

Evaluation metrics:
- Accuracy  
- Precision/Recall  
- ROC Curve & AUC  
- Feature importance  

### **4. Tools**
- Python  
- Pandas, NumPy, Matplotlib, Seaborn  
- scikit-learn  
- Jupyter Notebook  

---

## 📊 Results Visualization

### 🔥 Feature Importance
<p align="center">
  <img src="results/figures/feature_importance.png" width="600">
</p>

<details>
<summary>📈 View ROC Curve</summary>
<p align="center">
  <img src="results/figures/roc_curve.png" width="600">
</p>
</details>

<details>
<summary>🎨 Correlation Heatmap</summary>
<p align="center">
  <img src="results/figures/correlation_heatmap.png" width="600">
</p>
</details>

---

## ⚙️ Repository Structure
