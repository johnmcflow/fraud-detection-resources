"""
Basic Fraud Detection Example
Usage: python basic_fraud_detector.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_sample_data():
    """Generate sample fraud detection dataset"""
    np.random.seed(42)
    n_samples = 10000
    
    # Generate features
    data = {
        'amount': np.random.lognormal(3, 1, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'merchant_risk': np.random.uniform(0, 1, n_samples),
        'user_age_days': np.random.randint(1, 1000, n_samples),
        'device_score': np.random.uniform(0, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels (realistic fraud rate ~0.5%)
    fraud_probability = (
        0.001 + 
        0.01 * (df['amount'] > df['amount'].quantile(0.95)).astype(int) +
        0.005 * (df['merchant_risk'] > 0.8).astype(int) +
        0.002 * (df['hour'].isin([2, 3, 4])).astype(int)
    )
    
    df['is_fraud'] = np.random.binomial(1, fraud_probability)
    
    return df

def main():
    """Run basic fraud detection example"""
    print("Basic Fraud Detection Example")
    print("=" * 40)
    
    # Load data
    print("Loading sample data...")
    df = load_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3f}")
    
    # Prepare features
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle imbalance
    print("\n⚖️  Balancing dataset with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train model
    print("🤖 Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    print("\nResults:")
    print("-" * 20)
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top Features:")
    print(feature_importance.head())
    
    print("\nExample completed successfully!")
    print("Next steps: Try with real datasets from datasets/ folder")

if __name__ == "__main__":
    main()
