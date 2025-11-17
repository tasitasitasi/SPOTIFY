🎵 Predicting Spotify Track Popularity (2012–2021)
FAU ISC 4941 – Data Science Capstone Project

Team Members:
Nikita Belii · Tasianna Giordano · Anthony Gutierrez · Martin Gonzalez

📖 Overview

This project explores whether we can predict a song’s popularity on Spotify based on its audio features and metadata. Using a dataset of over 62,000 tracks released between 2012 and 2021, we trained machine learning models to identify which musical elements—like energy, danceability, loudness, and valence—most strongly influence popularity.

Spotify’s popularity score (0–100) reflects streaming frequency, recency, and user engagement. For this project, we converted it into a binary label (popular vs. not popular) to simplify the task into a classification problem.

🎯 Objective

Our main goal was to build a predictive model that could estimate how popular a song might become using only its audio features. Beyond prediction, we wanted to uncover meaningful insights about what makes music resonate with listeners in the modern streaming era.

💡 Key Findings

Dataset filtered to 62,015 tracks from 2012–2021 to align with Spotify’s modern metrics (post-“Follow” feature).

Gradient Boosting Classifier achieved around 75% accuracy, with balanced precision and recall.

Top predictors of popularity:

Energy

Danceability

Loudness

Release Year

Probability calibration improved interpretability—model probabilities reflect realistic likelihoods of success.

Results show that modern, high-energy tracks tend to perform better, aligning with current streaming trends.

🧠 Methodology

Data Source:

Spotify Dataset 1921–2020 (Kaggle)

Spotify Web API

Data Preparation:

Filtered tracks released after 2012

Cleaned missing values and normalized formats

Created binary label: is_popular (top 20% = popular)

Train/test split (80/20, stratified)

Modeling:

Algorithms tested: Linear Regression, Random Forest, Gradient Boosting

Evaluation metrics: Accuracy, Precision, Recall, ROC-AUC

Visualization: Correlation heatmaps, ROC curve, feature importance charts

Tools & Libraries:

Python (Pandas, NumPy, scikit-learn, Matplotlib, Seaborn)

Jupyter Notebook for development and visualization

📊 Results Visualization

Feature Correlation Heatmap showing relationships among key features.

ROC Curve indicating model performance with AUC > 0.8.

Feature Importance Chart highlighting energy, danceability, and loudness as dominant predictors.

(See /notebooks/visualizations/ for figures.)

🧩 Future Work

Integrate Spotify API for real-time song prediction using track IDs.

Incorporate genre and lyrical sentiment analysis.

Explore temporal shifts in popularity trends using yearly models.

Deploy a web-based dashboard where users can input a song and view its predicted popularity score.
