# Dataset Preprocessing Scripts

Scripts to prepare popular fraud detection datasets for machine learning models.

## Available Scripts

### Credit Card Datasets
- `creditcard_prep.py` - Credit Card Fraud Dataset preprocessing
- `ieee_cis_prep.py` - IEEE-CIS Fraud Detection preprocessing

### Graph Datasets  
- `elliptic_prep.py` - Bitcoin Elliptic dataset for GNN
- `paysim_prep.py` - PaySim synthetic dataset preprocessing

### Graph Conversion
- `create_graph_dataset.py` - Convert tabular data to graph format
- `graph_features.py` - Extract graph-based features

## Usage
```bash
python creditcard_prep.py --input creditcard.csv --output processed/
python create_graph_dataset.py --input transactions.csv --format pytorch
```

*Preprocessing scripts coming soon. See main README for dataset links.*
