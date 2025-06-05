# 🏠 Boston House Price Prediction

This repository contains my **Week 1 task** for the **ShadowFox AI/ML Internship**, where I built a regression model to predict Boston housing prices using machine learning techniques.

---

## ✅ Internship Pre-requisites Checklist

- ✅ LinkedIn profile updated with **"ShadowFox AIML Intern"** under Experience.
- ✅ GitHub repository named **ShadowFox**.
- ✅ Video explanation of the task posted on LinkedIn as **Proof of Work (POW)**.
- ✅ Task completed and documented as per guidelines.

---

## 📌 Project Overview

The goal of this project is to predict **median house prices in Boston** using features like crime rate, number of rooms, distance to work areas, etc. I implemented regression models and selected the best-performing one after evaluation and tuning.

---

## 🧰 Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-learn (Linear Regression, Decision Tree, Gradient Boosting, GridSearchCV)

---

## 📁 Dataset

- File: `HousingData.csv`
- Rows: 506
- Features: 13
- Target: `MEDV` (Median home price)

---

## 📊 Workflow Summary

1. **Data Cleaning**  
   - Filled missing values using **median** for 6 columns  
   - Capped outliers using **IQR method**

2. **Feature & Target Setup**  
   - Target: `MEDV`  
   - Features: 13 columns like `RM`, `LSTAT`, `CRIM`, etc.

3. **Model Training**  
   - Trained 3 models:
     - Linear Regression → R² = 0.7371
     - Decision Tree → R² = 0.7365
     - Gradient Boosting → R² = 0.8811

4. **Hyperparameter Tuning**  
   - Used `GridSearchCV` to tune Gradient Boosting  
   - Best parameters: `learning_rate=0.05`, `max_depth=4`, `n_estimators=100`

5. **Optimized Results**  
   - MSE: 5.3081  
   - R² Score: **0.8915**  
   - Accuracy improved by **8.73%**

6. **Feature Importance**  
   - Top Features:
     - `LSTAT`: 42%
     - `RM`: 37%
     - `DIS`, `CRIM`, `NOX`, etc.

---

## 📈 Final Outcome

- Best Model: **Optimized Gradient Boosting**
- Accuracy: **89.15%**
- MSE: **5.3081**

---



---

## 🏁 Conclusion

Successfully completed the beginner task for ShadowFox AIML Internship. Learned about data cleaning, model comparison, and hyperparameter tuning. Ready to move to intermediate-level projects!

---

### 🔖 Tags

`#ShadowFox #MachineLearning #Regression #Python #AIMLInternship #BostonHousing #ScikitLearn #GitHub #ProofOfWork`
