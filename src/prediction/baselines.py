import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def evaluate_baselines(data_path):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Define features and target.
    # Note: 'confidence' is a Stage 2 input, NOT a Stage 1 predictor.
    X = df.drop(columns=['target_score', 'confidence'])
    y = df['target_score']
    
    # 20% Held-out test set (Seed 42 for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set size: {len(X_train)} rows | Test set size: {len(X_test)} rows")
    
    results = {}
    
    def calculate_metrics(y_true, y_pred):
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }
        
    print("\nCalculating Baseline 1: Global Mean...")
    global_mean = y_train.mean()
    y_pred_mean = np.full_like(y_test, fill_value=global_mean, dtype=float)
    results['Global Mean'] = calculate_metrics(y_test, y_pred_mean)
    
    print("Calculating Baseline 2: Student's Own Average...")
    # Calculate each student's average score from the training data
    train_df = pd.concat([X_train, y_train], axis=1)
    student_means = train_df.groupby('student_id')['target_score'].mean()
    
    # Map their average to the test set (fallback to global mean if student is brand new)
    y_pred_student = X_test['student_id'].map(student_means).fillna(global_mean)
    results["Student Average"] = calculate_metrics(y_test, y_pred_student)
    
    print("Calculating Baseline 3: Linear Regression...")
    # Setup preprocessing: scale numeric features, one-hot encode the categorical 'subject'
    numeric_features = X_train.drop(columns=['student_id', 'subject']).columns.tolist()
    categorical_features = ['subject']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
        
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train and predict
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    results['Linear Regression'] = calculate_metrics(y_test, y_pred_lr)
    
    # Print the final formatted table
    results_df = pd.DataFrame(results).T
    print("\n" + "="*50)
    print("BASELINE RESULTS (HELD-OUT TEST SET)")
    print("="*50)
    print(results_df.round(3))
    print("="*50)

if __name__ == "__main__":
    data_file = os.path.join("data", "cleaned_features.csv")
    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
    else:
        evaluate_baselines(data_file)