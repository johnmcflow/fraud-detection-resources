"""
Real-Time Fraud Detection API Demo
Usage: python real_time_api_demo.py
Then visit: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn

# Mock model for demo (in production, load real model)
class MockFraudModel:
    def predict_proba(self, X):
        # Simple rule-based scoring for demo
        scores = []
        for _, row in X.iterrows():
            score = 0.1  # Base fraud probability
            
            # High amount = higher risk
            if row['amount'] > 1000:
                score += 0.3
                
            # Night transactions = higher risk  
            if row.get('hour', 12) in [2, 3, 4]:
                score += 0.2
                
            # New device = higher risk
            if row.get('device_age_days', 100) < 1:
                score += 0.4
                
            scores.append([1-score, score])
        return np.array(scores)

app = FastAPI(title="Fraud Detection API", version="1.0.0")
model = MockFraudModel()

class Transaction(BaseModel):
    user_id: str
    amount: float
    merchant_category: str = "unknown"
    hour: int = 12
    device_age_days: int = 100
    
class FraudResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    recommendation: str

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "model_version": "demo-1.0"
    }

@app.post("/predict", response_model=FraudResponse)
def predict_fraud(transaction: Transaction):
    """Predict fraud probability for a transaction"""
    
    # Convert to DataFrame for model
    df = pd.DataFrame([transaction.dict()])
    
    # Get prediction
    prob_scores = model.predict_proba(df)
    fraud_prob = prob_scores[0][1]
    
    # Determine risk level
    if fraud_prob > 0.7:
        risk_level = "HIGH"
        recommendation = "BLOCK transaction and review manually"
    elif fraud_prob > 0.3:
        risk_level = "MEDIUM" 
        recommendation = "Additional verification required"
    else:
        risk_level = "LOW"
        recommendation = "APPROVE transaction"
    
    return FraudResponse(
        transaction_id=f"tx_{hash(str(transaction.dict())) % 100000}",
        fraud_probability=round(fraud_prob, 3),
        risk_level=risk_level,
        recommendation=recommendation
    )

@app.post("/batch-predict")
def batch_predict(transactions: list[Transaction]):
    """Predict fraud for multiple transactions"""
    results = []
    for tx in transactions:
        result = predict_fraud(tx)
        results.append(result)
    return {"predictions": results}

if __name__ == "__main__":
    print("Starting Fraud Detection API...")
    print("API docs: http://localhost:8000/docs") 
    print("Health check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
