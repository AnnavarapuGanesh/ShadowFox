# 🏦 Loan Approval Prediction using Machine Learning

This project is part of my **Week 2 task** for the **ShadowFox AI/ML Internship (Intermediate Level)**. The goal is to build a robust classification model that predicts whether a loan application will be approved or not, based on the applicant’s financial, personal, and credit-related information.

---

## ✅ Internship Task Checklist

- ✅ GitHub repo: Uploaded in `ShadowFox/week_2_loan_approval_prediction`
- ✅ LinkedIn post with video explanation (Proof of Work)
- ✅ README with project documentation
- ✅ Visualizations and model comparison
- ✅ Followed internship formatting and submission rules

---

## 🧠 Problem Statement

Loan approval is a major task in the financial sector. This project uses **machine learning** to automate and enhance decision-making regarding loan approvals based on applicant details such as:

- Income and co-applicant income
- Employment and education status
- Loan amount and credit history
- Property area and dependents

---

## 📁 Dataset Information

- File: `loan_prediction.csv`
- Total records: 614
- Target variable: `Loan_Status` (Y = Approved, N = Denied)
- Source: Provided by ShadowFox [📎 Dataset Link](https://drive.google.com/drive/folders/18nheKtzhesFv_M6DB081dcmvphQXs7st)

---

## 🧰 Tools & Libraries Used

- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Plotly (if available)
- Scikit-learn: Random Forest, SVM, Logistic Regression, GridSearchCV
- Pipelines, Preprocessing, Evaluation metrics

---

## 🔄 Workflow Summary

### 1️⃣ Data Loading & Inspection
- Loaded `loan_prediction.csv` with shape `(614, 13)`
- Checked missing values and class imbalance

### 2️⃣ Data Preprocessing
- Dropped `Loan_ID`
- Filled missing values:
  - Categorical columns → **Mode**
  - Numerical columns → **Median**
- Handled outliers using **IQR method**

### 3️⃣ Exploratory Data Analysis (EDA)
- Pie and bar charts for:
  - Loan status, Gender, Education, Property Area
- Histograms for:
  - Applicant Income, Loan Amount
- Correlation matrix via heatmap

### 4️⃣ Feature Engineering
- Derived features:
  - `Total_Income`, `Loan_Income_Ratio`, `Income_per_Dependent`
  - Log-transformed: `ApplicantIncome`, `LoanAmount`

### 5️⃣ Model Building
Trained and evaluated 3 models:

| Model                | Accuracy | ROC AUC |
|---------------------|----------|---------|
| Logistic Regression | ~0.81    | ~0.84   |
| SVM                 | ~0.83    | ~0.85   |
| Random Forest       | **~0.85**| **~0.87** ✅ Best

> Used `train_test_split`, pipelines with ColumnTransformer, and cross-validation.

### 6️⃣ Model Optimization
- Used **GridSearchCV** to optimize **Random Forest**
- Tuned parameters:
  - `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- Final accuracy improved from `~0.85` → **~0.88**

---

## 📊 Results Summary

- ✅ Best model: **Optimized Random Forest**
- 🎯 Final Accuracy: **~0.88**
- 📈 ROC AUC: **~0.89**
- 🔍 Feature importance and ROC curves analyzed
- ✅ Confusion matrix and classification report plotted

---



## 🏁 Conclusion

- Learned how to structure a full machine learning pipeline
- Improved skills in EDA, feature engineering, model comparison, tuning, and evaluation
- Ready to take on more complex ML and AI challenges in upcoming tasks!

---

## 🔖 Tags

`#ShadowFox #AIMLInternship #LoanPrediction #MachineLearning #Python #RandomForest #SVM #Classification #GridSearch #ScikitLearn #GitHub #ProofOfWork`
