import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

print("All packages imported successfully!")
print("Starting Boston House Price Prediction Analysis...")

try:
    data = pd.read_csv('HousingData.csv')
    print("✅ Dataset loaded successfully!")
    print(f"Dataset shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())
except FileNotFoundError:
    print("❌ Error: HousingData.csv not found in current directory")
    print("Please make sure your dataset file is in the same folder as this script")
    print("Current directory should contain:")
    print("  - boston_housing_analysis.py")
    print("  - HousingData.csv")
    exit()

print("\n" + "="*50)
print("DATA EXPLORATION")
print("="*50)

print("\nDataset info:")
print(data.info())

print("\nMissing values check:")
missing_values = data.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print("\nHandling missing values...")
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if data[column].dtype in ['float64', 'int64']:
                data[column].fillna(data[column].median(), inplace=True)
                print(f"✅ Filled {column} with median")
    print("\nMissing values after cleaning:")
    print(data.isnull().sum())
else:
    print("✅ No missing values found!")


print("\nHandling outliers...")
def handle_outliers(df):
    df_cleaned = df.copy()
    outliers_info = {}
    
    for column in df_cleaned.select_dtypes(include=[np.number]).columns:
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        
        outliers_count = ((df_cleaned[column] < lower_bound) | (df_cleaned[column] > upper_bound)).sum()
        outliers_info[column] = outliers_count
        
        
        df_cleaned[column] = np.where(df_cleaned[column] < lower_bound, lower_bound, df_cleaned[column])
        df_cleaned[column] = np.where(df_cleaned[column] > upper_bound, upper_bound, df_cleaned[column])
    
    return df_cleaned, outliers_info

data_cleaned, outliers_info = handle_outliers(data)

print("Outliers found and capped:")
for col, count in outliers_info.items():
    if count > 0:
        print(f"  {col}: {count} outliers")


target_columns = ['MEDV', 'medv', 'price', 'target']
target_col = None

for col in target_columns:
    if col in data_cleaned.columns:
        target_col = col
        break

if target_col:
    X = data_cleaned.drop(target_col, axis=1)
    y = data_cleaned[target_col]
    print(f"✅ Using '{target_col}' as target variable")
else:
    
    X = data_cleaned.iloc[:, :-1]
    y = data_cleaned.iloc[:, -1]
    target_col = y.name
    print(f"✅ Using last column '{target_col}' as target variable")

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Data split completed:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\n🔄 Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ {model_name} completed:")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²:  {r2:.4f}")
    
    return model, mse, r2, y_pred


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

results = {}
predictions = {}

for name, model in models.items():
    trained_model, mse, r2, y_pred = train_and_evaluate_model(
        model, name, X_train, y_train, X_test, y_test
    )
    results[name] = {'model': trained_model, 'mse': mse, 'r2': r2}
    predictions[name] = y_pred


best_model_name = min(results.keys(), key=lambda x: results[x]['mse'])
print(f"\n🏆 Best performing model: {best_model_name}")


print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION")
print("="*60)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2]
}

print("🔍 Performing Grid Search for Gradient Boosting...")
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

print(f"✅ Grid Search completed!")
print(f"🎯 Best parameters: {gb_grid.best_params_}")
print(f"📈 Best CV score: {-gb_grid.best_score_:.4f}")


best_gb = gb_grid.best_estimator_
y_pred_best = best_gb.predict(X_test)
best_mse = mean_squared_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)

print(f"\n🚀 Optimized Gradient Boosting Results:")
print(f"   MSE: {best_mse:.4f}")
print(f"   R²:  {best_r2:.4f}")


print("\n" + "="*70)
print("🏁 FINAL RESULTS SUMMARY")
print("="*70)

final_results = {
    'Linear Regression': {'MSE': results['Linear Regression']['mse'], 'R²': results['Linear Regression']['r2']},
    'Decision Tree': {'MSE': results['Decision Tree']['mse'], 'R²': results['Decision Tree']['r2']},
    'Gradient Boosting': {'MSE': results['Gradient Boosting']['mse'], 'R²': results['Gradient Boosting']['r2']},
    'Optimized Gradient Boosting': {'MSE': best_mse, 'R²': best_r2}
}

for model_name, metrics in final_results.items():
    print(f"\n📊 {model_name}:")
    print(f"    MSE: {metrics['MSE']:.4f}")
    print(f"    R²:  {metrics['R²']:.4f}")


print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS")
print("="*50)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_gb.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Most Important Features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:12s}: {row['Importance']:.4f}")


original_gb_mse = results['Gradient Boosting']['mse']
improvement = ((original_gb_mse - best_mse) / original_gb_mse) * 100

print(f"\n📈 OPTIMIZATION RESULTS:")
print(f"   Original GB MSE: {original_gb_mse:.4f}")
print(f"   Optimized GB MSE: {best_mse:.4f}")
print(f"   Improvement: {improvement:.2f}%")

print(f"\n🎉 Analysis completed successfully!")
print(f"   Best model: Optimized Gradient Boosting")
print(f"   Final R² Score: {best_r2:.4f} ({best_r2*100:.1f}% variance explained)")
print(f"   Final MSE: {best_mse:.4f}")
