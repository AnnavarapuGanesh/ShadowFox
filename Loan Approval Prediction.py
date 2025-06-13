# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Try importing plotly, use fallback if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    print("✅ Plotly available - Using interactive visualizations")
except ImportError:
    print("⚠️ Plotly not available - Using matplotlib for visualizations")
    PLOTLY_AVAILABLE = False

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, accuracy_score, 
                           confusion_matrix, roc_auc_score, roc_curve)

# Load and inspect the dataset
def load_and_inspect_data():
    """Load dataset and perform initial inspection"""
    # Your specific file path
    file_path = r"C:\Users\ANNAVARAPU GANESH\OneDrive\Desktop\Loan Approval Prediction System\loan_prediction.csv"
    
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully!")
        print("Dataset Shape:", df.shape)
        print("\nColumn Information:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nTarget Variable Distribution:")
        print(df['Loan_Status'].value_counts())
        return df
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {file_path}")
        print("Please check if the file exists at this location.")
        return None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return None

# Data Preprocessing
def preprocess_data(df):
    """Comprehensive data preprocessing"""
    if df is None:
        return None
        
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Drop irrelevant columns
    if 'Loan_ID' in data.columns:
        data.drop(columns=['Loan_ID'], inplace=True)
        print("✅ Dropped Loan_ID column")
    
    # Handle missing values
    print("🔄 Handling missing values...")
    
    # Categorical columns - fill with mode
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 
                       'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            mode_value = data[col].mode()[0] if not data[col].mode().empty else 'Unknown'
            missing_count = data[col].isnull().sum()
            data[col].fillna(mode_value, inplace=True)
            print(f"  - Filled {missing_count} missing values in {col} with: {mode_value}")
    
    # Numerical columns - fill with median
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                     'Loan_Amount_Term', 'Credit_History']
    
    for col in numerical_cols:
        if col in data.columns and data[col].isnull().sum() > 0:
            median_value = data[col].median()
            missing_count = data[col].isnull().sum()
            data[col].fillna(median_value, inplace=True)
            print(f"  - Filled {missing_count} missing values in {col} with median: {median_value}")
    
    # Handle outliers using IQR method
    def handle_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        # Count outliers before treatment
        outliers_count = ((df[column] < lower_limit) | (df[column] > upper_limit)).sum()
        
        # Cap outliers
        df[column] = np.where(df[column] < lower_limit, lower_limit, df[column])
        df[column] = np.where(df[column] > upper_limit, upper_limit, df[column])
        
        if outliers_count > 0:
            print(f"  - Handled {outliers_count} outliers in {column}")
        
        return df
    
    # Apply outlier handling to key numerical columns
    print("🔄 Handling outliers...")
    outlier_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in outlier_cols:
        if col in data.columns:
            data = handle_outliers(data, col)
    
    print("✅ Data preprocessing completed!")
    print(f"Final dataset shape: {data.shape}")
    return data

# Exploratory Data Analysis
def perform_eda(df):
    """Comprehensive Exploratory Data Analysis"""
    print("📊 Performing Exploratory Data Analysis...")
    
    # Basic statistics
    print("\n📈 Descriptive Statistics:")
    print(df.describe())
    
    if PLOTLY_AVAILABLE:
        # Interactive Plotly visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loan Status Distribution', 'Gender Distribution',
                           'Education Level', 'Property Area'),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Loan Status
        loan_counts = df['Loan_Status'].value_counts()
        fig.add_trace(go.Pie(labels=loan_counts.index, values=loan_counts.values,
                            name="Loan Status"), row=1, col=1)
        
        # Gender Distribution
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            fig.add_trace(go.Pie(labels=gender_counts.index, values=gender_counts.values,
                                name="Gender"), row=1, col=2)
        
        # Education Level
        if 'Education' in df.columns:
            education_counts = df['Education'].value_counts()
            fig.add_trace(go.Bar(x=education_counts.index, y=education_counts.values,
                                name="Education"), row=2, col=1)
        
        # Property Area
        if 'Property_Area' in df.columns:
            property_counts = df['Property_Area'].value_counts()
            fig.add_trace(go.Bar(x=property_counts.index, y=property_counts.values,
                                name="Property Area"), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Loan Dataset Overview")
        fig.show()
    else:
        # Matplotlib fallback
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Loan Dataset Analysis', fontsize=16)
        
        # Loan Status Distribution
        loan_counts = df['Loan_Status'].value_counts()
        axes[0,0].pie(loan_counts.values, labels=loan_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Loan Status Distribution')
        
        # Gender Distribution
        if 'Gender' in df.columns:
            gender_counts = df['Gender'].value_counts()
            axes[0,1].bar(gender_counts.index, gender_counts.values, color='skyblue')
            axes[0,1].set_title('Gender Distribution')
            axes[0,1].set_xlabel('Gender')
            axes[0,1].set_ylabel('Count')
        
        # Education Level
        if 'Education' in df.columns:
            education_counts = df['Education'].value_counts()
            axes[0,2].bar(education_counts.index, education_counts.values, color='lightgreen')
            axes[0,2].set_title('Education Level')
            axes[0,2].set_xlabel('Education')
            axes[0,2].set_ylabel('Count')
        
        # Income Distribution
        if 'ApplicantIncome' in df.columns:
            axes[1,0].hist(df['ApplicantIncome'], bins=30, alpha=0.7, color='orange')
            axes[1,0].set_title('Applicant Income Distribution')
            axes[1,0].set_xlabel('Income')
            axes[1,0].set_ylabel('Frequency')
        
        # Loan Amount Distribution
        if 'LoanAmount' in df.columns:
            axes[1,1].hist(df['LoanAmount'].dropna(), bins=30, alpha=0.7, color='purple')
            axes[1,1].set_title('Loan Amount Distribution')
            axes[1,1].set_xlabel('Loan Amount')
            axes[1,1].set_ylabel('Frequency')
        
        # Loan Status by Property Area
        if 'Property_Area' in df.columns:
            loan_property = pd.crosstab(df['Property_Area'], df['Loan_Status'])
            loan_property.plot(kind='bar', ax=axes[1,2], color=['red', 'green'])
            axes[1,2].set_title('Loan Status by Property Area')
            axes[1,2].set_xlabel('Property Area')
            axes[1,2].legend(['Denied', 'Approved'])
            axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    # Additional Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Applicant Income Distribution
    if 'ApplicantIncome' in df.columns:
        axes[0,0].hist(df['ApplicantIncome'], bins=30, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Applicant Income Distribution')
        axes[0,0].set_xlabel('Income')
    
    # Loan Amount Distribution
    if 'LoanAmount' in df.columns:
        axes[0,1].hist(df['LoanAmount'].dropna(), bins=30, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('Loan Amount Distribution')
        axes[0,1].set_xlabel('Loan Amount')
    
    # Loan Status by Gender
    if 'Gender' in df.columns:
        loan_gender = pd.crosstab(df['Gender'], df['Loan_Status'])
        loan_gender.plot(kind='bar', ax=axes[1,0], color=['red', 'green'])
        axes[1,0].set_title('Loan Status by Gender')
        axes[1,0].set_xlabel('Gender')
        axes[1,0].legend(['Denied', 'Approved'])
        axes[1,0].tick_params(axis='x', rotation=0)
    
    # Loan Status by Education
    if 'Education' in df.columns:
        loan_education = pd.crosstab(df['Education'], df['Loan_Status'])
        loan_education.plot(kind='bar', ax=axes[1,1], color=['red', 'green'])
        axes[1,1].set_title('Loan Status by Education')
        axes[1,1].set_xlabel('Education')
        axes[1,1].legend(['Denied', 'Approved'])
        axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation Analysis
    print("🔗 Creating correlation matrix...")
    df_numeric = df.copy()
    
    # Encode categorical variables for correlation analysis
    le = LabelEncoder()
    categorical_columns = df_numeric.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
    
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Feature Engineering
def feature_engineering(df):
    """Advanced feature engineering"""
    print("🔧 Performing feature engineering...")
    data = df.copy()
    
    # Create new features
    print("  - Creating derived features...")
    
    # Total Income
    if 'ApplicantIncome' in data.columns and 'CoapplicantIncome' in data.columns:
        data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
        print("    ✓ Total_Income created")
    
    # Loan Amount to Income Ratio
    if 'LoanAmount' in data.columns and 'Total_Income' in data.columns:
        data['Loan_Income_Ratio'] = data['LoanAmount'] / data['Total_Income']
        data['Loan_Income_Ratio'].replace([np.inf, -np.inf], 0, inplace=True)
        print("    ✓ Loan_Income_Ratio created")
    
    # Income per dependent
    if 'Dependents' in data.columns and 'Total_Income' in data.columns:
        data['Dependents_numeric'] = data['Dependents'].replace('3+', '3').astype(float)
        data['Income_per_Dependent'] = data['Total_Income'] / (data['Dependents_numeric'] + 1)
        print("    ✓ Income_per_Dependent created")
    
    # Log transformation for skewed features
    if 'ApplicantIncome' in data.columns:
        data['Log_ApplicantIncome'] = np.log1p(data['ApplicantIncome'])
        print("    ✓ Log_ApplicantIncome created")
    
    if 'LoanAmount' in data.columns:
        data['Log_LoanAmount'] = np.log1p(data['LoanAmount'])
        print("    ✓ Log_LoanAmount created")
    
    print("✅ Feature engineering completed!")
    print(f"New dataset shape: {data.shape}")
    
    return data

# Model Training and Evaluation
def train_and_evaluate_models(df):
    """Train multiple models and evaluate performance"""
    print("🤖 Training and evaluating models...")
    
    # Prepare features and target
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Identify feature types
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {len(numerical_features)} features")
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])
    
    # Define models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  ✅ Accuracy: {accuracy:.4f}")
        print(f"  ✅ ROC AUC: {roc_auc:.4f}")
        print(f"  ✅ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print(f"\n📊 Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Denied', 'Approved'], 
                   yticklabels=['Denied', 'Approved'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    return results, X_test, y_test

# Model Optimization
def optimize_best_model(df):
    """Hyperparameter tuning for the best performing model"""
    print("🎯 Optimizing best model...")
    
    # Prepare data
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ])
    
    # Random Forest hyperparameter tuning
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Grid search
    print("🔍 Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Test accuracy with optimized model: {accuracy:.4f}")
    
    return best_model, X_test, y_test, y_pred

# Visualization of Results
def visualize_results(results, X_test, y_test):
    """Visualize model performance"""
    print("📈 Creating performance visualizations...")
    
    # Model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    roc_aucs = [results[name]['roc_auc'] for name in model_names]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy comparison
    bars1 = axes[0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    # ROC AUC comparison
    bars2 = axes[1].bar(model_names, roc_aucs, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1].set_title('Model ROC AUC Comparison')
    axes[1].set_ylabel('ROC AUC')
    axes[1].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(roc_aucs):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    
    for name in model_names:
        y_pred_proba = results[name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = results[name]['roc_auc']
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Main execution function
def main():
    """Main execution pipeline"""
    
    print("=" * 60)
    print("🚀 LOAN APPROVAL PREDICTION SYSTEM")
    print("=" * 60)
    
    # Load data
    print("\n1️⃣ Loading and inspecting data...")
    df = load_and_inspect_data()
    
    if df is None:
        print("❌ Cannot proceed without dataset. Please check file path.")
        return None, None
    
    # Preprocess data
    print("\n2️⃣ Preprocessing data...")
    df_processed = preprocess_data(df)
    
    if df_processed is None:
        print("❌ Data preprocessing failed.")
        return None, None
    
    # Exploratory Data Analysis
    print("\n3️⃣ Performing Exploratory Data Analysis...")
    perform_eda(df_processed)
    
    # Feature Engineering
    print("\n4️⃣ Feature Engineering...")
    df_engineered = feature_engineering(df_processed)
    
    # Train and evaluate models
    print("\n5️⃣ Training and evaluating models...")
    results, X_test, y_test = train_and_evaluate_models(df_engineered)
    
    # Visualize results
    print("\n6️⃣ Visualizing results...")
    visualize_results(results, X_test, y_test)
    
    # Optimize best model
    print("\n7️⃣ Optimizing best model...")
    best_model, X_test_opt, y_test_opt, y_pred_opt = optimize_best_model(df_engineered)
    
    # Final Results
    print("\n" + "=" * 60)
    print("🎉 FINAL RESULTS")
    print("=" * 60)
    
    # Find best performing model from initial comparison
    best_initial_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_initial_accuracy = results[best_initial_model]['accuracy']
    optimized_accuracy = accuracy_score(y_test_opt, y_pred_opt)
    
    print(f"📊 Model Performance Summary:")
    print(f"  • Best Initial Model: {best_initial_model}")
    print(f"  • Initial Best Accuracy: {best_initial_accuracy:.4f}")
    print(f"  • Optimized Model Accuracy: {optimized_accuracy:.4f}")
    print(f"  • Improvement: {optimized_accuracy - best_initial_accuracy:.4f}")
    
    print(f"\n✅ Model training and evaluation completed successfully!")
    print(f"🎯 Final optimized model achieved {optimized_accuracy:.4f} accuracy on test set")
    
    return best_model, df_engineered

# Execute the complete pipeline
if __name__ == "__main__":
    print("Starting Loan Approval Prediction System...")
    best_model, processed_data = main()
    
    if best_model is not None:
        print("\n🎊 System execution completed successfully!")
        print("📁 All visualizations have been displayed.")
        print("💾 Best model is ready for predictions.")
    else:
        print("\n❌ System execution failed. Please check the errors above.")
