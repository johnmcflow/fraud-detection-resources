"""
Feature Engineering Utilities for Fraud Detection
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

class FraudFeatureEngine:
    def __init__(self):
        self.encoders = {}
        self.user_stats = {}
    
    def create_temporal_features(self, df, timestamp_col='timestamp'):
        """Create time-based features"""
        if timestamp_col not in df.columns:
            return df
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        return df
    
    def create_velocity_features(self, df, user_col='user_id', timestamp_col='timestamp', windows=[1, 5, 15, 60]):
        """Create transaction velocity features"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values([user_col, timestamp_col])
        
        for window in windows:
            window_str = f'{window}min'
            col_name = f'tx_count_{window}m'
            
            df[col_name] = df.groupby(user_col)[timestamp_col].rolling(
                window=window_str, min_periods=1
            ).count().reset_index(level=0, drop=True)
        
        return df
    
    def create_amount_features(self, df, amount_col='amount', user_col='user_id'):
        """Create amount-based features"""
        df = df.copy()
        
        # Log transform
        df['amount_log'] = np.log1p(df[amount_col])
        
        # User-level statistics
        user_stats = df.groupby(user_col)[amount_col].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).add_prefix('user_amount_')
        
        df = df.merge(user_stats, left_on=user_col, right_index=True, how='left')
        
        # Amount deviation from user norm
        df['amount_zscore'] = (df[amount_col] - df['user_amount_mean']) / (df['user_amount_std'] + 1e-6)
        
        return df
    
    def create_categorical_features(self, df, categorical_cols):
        """Encode categorical features"""
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def create_risk_features(self, df, high_risk_merchants=None, high_risk_hours=None):
        """Create risk-based features"""
        df = df.copy()
        
        if high_risk_merchants is None:
            high_risk_merchants = ['online', 'atm', 'gas_station']
        
        if high_risk_hours is None:
            high_risk_hours = [2, 3, 4, 23]
        
        # Merchant risk
        if 'merchant_category' in df.columns:
            df['is_high_risk_merchant'] = df['merchant_category'].isin(high_risk_merchants).astype(int)
        
        # Time risk
        if 'hour' in df.columns:
            df['is_high_risk_hour'] = df['hour'].isin(high_risk_hours).astype(int)
        
        return df

def engineer_features(df, config=None):
    """Main feature engineering pipeline"""
    if config is None:
        config = {
            'timestamp_col': 'timestamp',
            'user_col': 'user_id', 
            'amount_col': 'amount',
            'categorical_cols': ['merchant_category', 'device_type'],
            'velocity_windows': [1, 5, 15, 60]
        }
    
    engine = FraudFeatureEngine()
    
    # Apply feature engineering steps
    df = engine.create_temporal_features(df, config['timestamp_col'])
    df = engine.create_velocity_features(df, config['user_col'], config['timestamp_col'], config['velocity_windows'])
    df = engine.create_amount_features(df, config['amount_col'], config['user_col'])
    df = engine.create_categorical_features(df, config['categorical_cols'])
    df = engine.create_risk_features(df)
    
    return df

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'user_id': ['user_1', 'user_1', 'user_2', 'user_2'],
        'amount': [100, 250, 50, 1000],
        'timestamp': ['2024-01-01 14:30:00', '2024-01-01 15:45:00', 
                     '2024-01-01 02:15:00', '2024-01-01 02:20:00'],
        'merchant_category': ['grocery', 'restaurant', 'online', 'atm']
    }
    
    df = pd.DataFrame(sample_data)
    df_features = engineer_features(df)
    
    print("Feature engineering example:")
    print(df_features.columns.tolist())
