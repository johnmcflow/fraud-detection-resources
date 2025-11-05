"""
Tests for basic fraud detector
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code', 'examples'))

def test_basic_import():
    """Test that basic detector can be imported"""
    try:
        import basic_fraud_detector
        assert True
    except ImportError:
        assert False, "Cannot import basic_fraud_detector"

def test_load_sample_data():
    """Test sample data generation"""
    from basic_fraud_detector import load_sample_data
    
    df = load_sample_data()
    assert len(df) == 10000
    assert 'is_fraud' in df.columns
    assert df['is_fraud'].mean() < 0.1  # Reasonable fraud rate
    
def test_data_quality():
    """Test generated data quality"""
    from basic_fraud_detector import load_sample_data
    
    df = load_sample_data()
    
    # Check no missing values
    assert df.isnull().sum().sum() == 0
    
    # Check reasonable ranges
    assert df['amount'].min() > 0
    assert df['hour'].min() >= 0 and df['hour'].max() <= 23
    assert df['merchant_risk'].min() >= 0 and df['merchant_risk'].max() <= 1

if __name__ == "__main__":
    print("Running tests...")
    test_basic_import()
    test_load_sample_data() 
    test_data_quality()
    print("✅ All tests passed!")
