import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv("data/cleaned_features.csv")
    
    # Define features and target. Confidence is excluded.
    X = df.drop(columns=['target_score', 'confidence'])
    y = df['target_score']
    
    # 20% Held-out test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessing
    numeric_features = X_train.drop(columns=['student_id', 'subject']).columns.tolist()
    categorical_features = ['subject']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
        
    print("Building MLP Pipeline with recommended defaults...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', mlp)
    ])
    
    print("Training model (this might take a few moments)...")
    pipeline.fit(X_train, y_train)
    
    print(f"Training complete. Early stopping triggered at epoch: {mlp.n_iter_}")
    
    # --- Step 6: Honest Test Evaluation ---
    y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MLP RESULTS (HELD-OUT TEST SET)")
    print("="*50)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    print("="*50)
    
    # Diagnostics: The 20 worst predictions
    errors = np.abs(y_test - y_pred)
    worst_indices = errors.nlargest(20).index
    print("\nError Analysis - Top 5 worst predictions:")
    for idx in worst_indices[:5]:
        print(f"True: {y_test.loc[idx]:.1f}, Pred: {y_pred.loc[idx]:.1f}, Diff: {errors.loc[idx]:.1f} | Subj: {df.loc[idx, 'subject']}")
        
    # --- Export Report Figures ---
    os.makedirs(os.path.join("docs", "figures"), exist_ok=True)
    
    # Plot 1: Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_, label='Training Loss')
    if mlp.early_stopping:
        plt.plot(mlp.validation_scores_, label='Validation Score (R2)')
    plt.title('MLP Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Score')
    plt.legend()
    plt.savefig(os.path.join("docs", "figures", "loss_curve.png"))
    plt.close()
    
    # Plot 2: Pred vs Actual
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') # Diagonal line
    plt.title('Predicted vs Actual Score')
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.savefig(os.path.join("docs", "figures", "pred_vs_actual.png"))
    plt.close()
    
    # Plot 3: Residuals
    plt.figure(figsize=(8, 5))
    plt.hist(y_test - y_pred, bins=50, edgecolor='k')
    plt.title('Residuals Histogram')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join("docs", "figures", "residuals.png"))
    plt.close()
    
    # --- Step 8: Persist the Pipeline ---
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "predictor_v1.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model and scaler to {model_path}")
    print("Saved all presentation figures to docs/figures/")

if __name__ == "__main__":
    train_and_evaluate()