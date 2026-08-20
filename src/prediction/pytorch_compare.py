import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define the PyTorch Model (exactly matching the scikit-learn architecture)
class StudentScoreMLP(nn.Module):
    def __init__(self, input_size):
        super(StudentScoreMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def run_pytorch_comparison():
    print("Loading data for PyTorch comparison...")
    df = pd.read_csv("data/cleaned_features.csv")
    
    X = df.drop(columns=['target_score', 'confidence'])
    y = df['target_score']
    
    # Same 80/20 Split (Seed 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Same Preprocessing
    numeric_features = X_train.drop(columns=['student_id', 'subject']).columns.tolist()
    categorical_features = ['subject']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    
    # Hardware Acceleration Check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize Model, Loss, and Optimizer
    input_size = X_train_processed.shape[1]
    model = StudentScoreMLP(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    
    # Training Loop (equivalent to scikit-learn's max_iter=1000)
    epochs = 1000
    print(f"Training PyTorch model for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    # Evaluation
    model.eval()
    X_test_tensor = X_test_tensor.to(device)
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()
        
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n" + "="*50)
    print("PYTORCH RESULTS (HELD-OUT TEST SET)")
    print("="*50)
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    print("="*50)

if __name__ == "__main__":
    run_pytorch_comparison()