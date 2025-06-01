# 🫀 Cardiovascular Disease Risk Prediction

This project explores the development of predictive models for identifying individuals at risk of cardiovascular disease using real-world health survey data. Through comprehensive exploratory analysis, feature engineering, and the application of multiple machine learning algorithms, the project aims to deliver interpretable and effective models for health-related prediction tasks.

## 📊 Dataset

The dataset was sourced from Kaggle:  
🔗 https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset

It contains health, lifestyle, and demographic information about individuals in the U.S., including variables such as:
- Age, sex, BMI
- Physical activity, smoking and alcohol habits
- Self-reported general health
- Diagnosis of diseases (diabetes, cancer, depression, arthritis, etc.)

The target variable is **Heart Disease** (binary classification: at risk or not).

---

## 🎯 Objectives

- Conduct detailed exploratory data analysis (EDA)
- Handle class imbalance using SMOTE and Tomek Links
- Apply feature engineering and scaling techniques
- Train, tune and compare multiple machine learning models
- Evaluate performance using F1-score and recall (for medical context)

---

## 📌 Methodology Overview

### 📈 Exploratory Data Analysis (EDA)
- Univariate and bivariate analysis with respect to the target
- Analysis of variable distributions, outliers, and correlations
- Key insights into lifestyle vs. heart disease risk

### 🛠 Feature Engineering
- Encoding categorical variables (Ordinal Encoding)
- Feature scaling (MinMaxScaler)
- Handling imbalanced data using **SMOTE** + **Tomek Links**
- Balancing classes from 8%/92% to ~50%/50%

### 🤖 Models Applied
Each model was tuned using grid search or parameter variation:
- **Naive Bayes**
- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **XGBoost Classifier**
- **AdaBoost**
- **Neural Networks (MLPClassifier)**

---

## 📈 Performance Comparison

| Model              | F1-score (Class 1) | Recall (Class 1) |
|-------------------|--------------------|------------------|
| Naive Bayes       | 0.28               | 0.75             |
| Logistic Regression | 0.30             | 0.77             |
| Random Forest     | **0.35**           | 0.91             |
| KNN               | **0.35**           | **0.94**         |
| XGBoost           | 0.34               | 0.85             |
| AdaBoost          | 0.31               | 0.35             |
| Neural Network    | 0.32               | 0.82             |

📌 **Key Insight**:  
- **KNN** had the best recall for the minority class (heart disease cases).
- **Random Forest** offered excellent recall with a balanced trade-off.
- **XGBoost** provided consistent results and a good precision-recall balance.

---

## 📚 Conclusion

- Proper preprocessing (resampling, scaling, encoding) is essential in medical ML tasks.
- Class imbalance severely impacts model performance and must be addressed.
- The combination of Random Forest and KNN produced the most medically useful results.
- F1-score and Recall are crucial metrics in healthcare prediction, where false negatives are costly.

---

## 🧠 Technologies Used

- Python (pandas, scikit-learn, XGBoost, imbalanced-learn)
- Jupyter Notebook
- Matplotlib & Seaborn for visualization

---

## 📁 Files

- `practica_100518506-CMG.ipynb`: Full notebook with data analysis and model training
- You can open it in Google Colab or Jupyter locally to explore the analysis.

---

> ✍️ Developed as part of an academic assignment in data analytics and predictive modeling.
