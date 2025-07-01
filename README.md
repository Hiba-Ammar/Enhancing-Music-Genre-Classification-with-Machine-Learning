# Enhancing-Music-Genre-Classification-with-Machine-Learning
This project focuses on building a robust machine learning pipeline to classify music tracks into their respective genres using structured audio features. The dataset includes 18 audio features per track, and the model aims to accurately assign one of several genre classes such as Rock, Pop, HipHop, Instrumental, Country, and more.

📌 Project Objective
To develop, train, and optimize machine learning models to predict music genres based on audio features with the goal of achieving high accuracy, precision, recall, and F1-score.

📂 Dataset
The dataset used in this project includes:

Train: 14,396 tracks with 18 audio features and labeled genres.

Test: 3,600 tracks with the same features (used for final predictions).

Source: Kaggle Competition: Music Genre Classification 2024

Each track includes:

Audio metrics like danceability, energy, acousticness, valence, etc.

Temporal and structural features like tempo, duration, key, and time_signature.

🧠 Machine Learning Pipeline
1. 🧼 Data Preprocessing
Handling missing values via mean imputation (Popularity, key, instrumentalness)

Label encoding for the target Class

Feature engineering:

danceability_energy = danceability × energy

loudness_valence = loudness × valence

log_duration = log(duration)

Feature binning (e.g. tempo → slow, medium, fast)

One-Hot Encoding for categorical features (tempo_bin, key)

Feature scaling using StandardScaler

2. 📊 Exploratory Data Analysis
Genre distribution using countplots

Correlation matrix of numerical features

Distribution and boxplot/violinplot comparisons by genre

3. ⚙️ Model Training & Evaluation
Models Evaluated:

Logistic Regression

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Gradient Boosting

XGBoost (final model)

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Best Results:

📈 XGBoost with tuned hyperparameters (GridSearchCV)

Best macro F1-score: 0.552

Best genres identified: HipHop, Blues, Bollywood

🧪 Hyperparameter Tuning
Tuned the final model (XGBoost) with:


📁 Output
Final predictions stored in sub.csv with format:

csv
Id, Class
1, Rock
2, Indie
...
🔍 Key Insights
Genre classes are moderately imbalanced.

Energy, valence, and danceability are highly influential features.

XGBoost outperformed other models, especially with SMOTE and feature engineering.

Some genres (e.g., Acoustic/Folk, Bollywood) benefit significantly from engineered features.

