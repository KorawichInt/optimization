import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv("AAPL.csv")
    
    # Separate features and target
    X = data[['Open', 'High', 'Low', 'Volume']]  # Example features
    y = data['Close']  # Target variable
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
