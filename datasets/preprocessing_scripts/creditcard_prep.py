"""
Credit Card Fraud Dataset Preprocessor
Downloads and prepares the famous Credit Card Fraud Detection dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def download_and_prep_creditcard(output_dir="processed"):
    """
    Prepare Credit Card Fraud dataset
    Note: You need to download from Kaggle first
    """
    print("Credit Card Fraud Dataset Preprocessor")
    print("=" * 45)
    
    # Check if raw data exists
    if not os.path.exists("creditcard.csv"):
        print("Error: creditcard.csv not found!")
        print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place in datasets/ folder")
        return
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv("creditcard.csv")
    
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['Class'].mean():.4f}")
    
    # Basic preprocessing
    print("Preprocessing...")
    
    # Add time-based features
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Day'] = df['Time'] // (24 * 3600)
    
    # Scale amount
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    
    # Create feature matrix
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Hour']
    X = df[feature_cols]
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False) 
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    print(f"Processed data saved to {output_dir}/")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    download_and_prep_creditcard()
