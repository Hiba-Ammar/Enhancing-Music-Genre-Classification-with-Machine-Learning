# 🎵 Music Genre Classification with Machine Learning

This project focuses on building a robust machine learning pipeline to classify music tracks into their respective genres using structured audio features. Each track includes 18 numerical features, such as danceability, tempo, and energy. The goal is to accurately assign each track one of several genre labels like **Rock**, **Pop**, **HipHop**, **Instrumental**, **Country**, and more.

---

## 📌 Project Objective

**To develop, train, and optimize machine learning models that classify music genres based on extracted audio features, aiming for high accuracy, precision, recall, and F1-score.**

---

## 📂 Dataset

- **Train**: 14,396 tracks with 18 features and labeled genres  
- **Test**: 3,600 tracks for prediction (unlabeled)  
- **Source**: [Kaggle Competition - Music Genre Classification 2024](https://www.kaggle.com/competitions/music-genre-classification-2024)

### 🔢 Features Include

- Audio characteristics: `danceability`, `energy`, `acousticness`, `valence`, `loudness`
- Temporal / structural features: `tempo`, `duration`, `key`, `time_signature`, `instrumentalness`

---

## 🧠 Machine Learning Pipeline

### 🧼 Data Preprocessing

- Mean imputation of missing values (`Popularity`, `Key`, `Instrumentalness`)
- Label encoding for `Class` (target)
- **Feature engineering:**
  - `danceability_energy` = danceability × energy
  - `loudness_valence` = loudness × valence
  - `log_duration` = log(duration)
- Tempo binning (`slow`, `medium`, `fast`)
- One-hot encoding for `tempo_bin`, `key`
- Feature scaling using `StandardScaler`

---

## 📊 Exploratory Data Analysis (EDA)

- Genre distribution via count plots
- Correlation matrix between audio features
- Boxplots and violin plots by genre

---

## ⚙️ Model Training & Evaluation

### ✅ Models Evaluated

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- **XGBoost** (final chosen model)

### 📏 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix

### 🔥 Best Results

- **XGBoost** with tuned hyperparameters via `GridSearchCV`
- **Macro F1-score**: `0.552`
- **Best genres identified**: HipHop, Blues, Bollywood

---

## 🧪 Hyperparameter Tuning

```python
params = {
    'xgb__learning_rate': [0.1, 0.01],
    'xgb__max_depth': [3, 5, 7],
    'xgb__reg_lambda': [1, 1.5, 2],
    'xgb__gamma': [1, 2],
}
Class imbalance handled using SMOTE

📁 Output
Final predictions saved in sub.csv with the format:

csv

Id,Class
1,Rock
2,Indie
...
🔍 Key Insights
Dataset contains moderate class imbalance

Energy, valence, and danceability are top predictive features

XGBoost + SMOTE + feature engineering gave the best results

Minor genres like Acoustic/Folk and Bollywood benefited most from feature engineering

🧩 Future Enhancements
Use spectrograms with CNNs (deep learning)

Try ensemble stacking or voting classifiers

Improve class balance via smarter augmentation

Include audio metadata (e.g., lyrics, mood)
