"""
Basic Graph-based Fraud Detection Example
Usage: python graph_fraud_basic.py
"""
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def create_sample_transaction_graph():
    """Create a sample transaction network"""
    np.random.seed(42)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add user nodes
    users = [f"user_{i}" for i in range(50)]
    merchants = [f"merchant_{i}" for i in range(20)]
    
    # Add nodes with attributes
    for user in users:
        G.add_node(user, node_type='user', risk_score=np.random.uniform(0, 1))
    
    for merchant in merchants:
        G.add_node(merchant, node_type='merchant', category=np.random.choice(['grocery', 'gas', 'restaurant', 'online']))
    
    # Add transaction edges
    n_transactions = 500
    for _ in range(n_transactions):
        user = np.random.choice(users)
        merchant = np.random.choice(merchants)
        amount = np.random.lognormal(3, 1)
        
        # Higher chance of fraud for high-risk users and large amounts
        user_risk = G.nodes[user]['risk_score']
        fraud_prob = 0.01 + 0.1 * user_risk + 0.05 * (amount > 100)
        is_fraud = np.random.binomial(1, min(fraud_prob, 0.3))
        
        G.add_edge(user, merchant, amount=amount, is_fraud=is_fraud)
    
    return G

def analyze_fraud_patterns(G):
    """Analyze fraud patterns in the transaction graph"""
    print("🕸️  Graph-based Fraud Analysis")
    print("=" * 40)
    
    # Basic graph statistics
    print(f"Graph Statistics:")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    print(f"   Density: {nx.density(G):.4f}")
    
    # Fraud statistics
    fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d['is_fraud'] == 1]
    total_edges = G.number_of_edges()
    fraud_rate = len(fraud_edges) / total_edges
    
    print(f"\nFraud Statistics:")
    print(f"   Total transactions: {total_edges}")
    print(f"   Fraudulent transactions: {len(fraud_edges)}")
    print(f"   Fraud rate: {fraud_rate:.3f}")
    
    # Node-level analysis
    user_fraud_counts = defaultdict(int)
    merchant_fraud_counts = defaultdict(int)
    
    for u, v, d in G.edges(data=True):
        if d['is_fraud'] == 1:
            if G.nodes[u]['node_type'] == 'user':
                user_fraud_counts[u] += 1
            if G.nodes[v]['node_type'] == 'merchant':
                merchant_fraud_counts[v] += 1
    
    # Top fraudulent users
    top_fraud_users = sorted(user_fraud_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop Fraudulent Users:")
    for user, count in top_fraud_users:
        risk_score = G.nodes[user]['risk_score']
        print(f"   {user}: {count} fraud transactions (risk: {risk_score:.3f})")
    
    # Suspicious merchant analysis
    merchant_stats = []
    for merchant in [n for n in G.nodes() if G.nodes[n]['node_type'] == 'merchant']:
        incoming_edges = G.in_edges(merchant, data=True)
        if len(incoming_edges) > 0:
            fraud_count = sum(1 for _, _, d in incoming_edges if d['is_fraud'] == 1)
            total_count = len(incoming_edges)
            fraud_ratio = fraud_count / total_count if total_count > 0 else 0
            merchant_stats.append((merchant, fraud_ratio, total_count))
    
    # Sort by fraud ratio
    merchant_stats.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏪 Suspicious Merchants:")
    for merchant, fraud_ratio, total_trans in merchant_stats[:5]:
        category = G.nodes[merchant]['category']
        print(f"   {merchant} ({category}): {fraud_ratio:.3f} fraud ratio ({total_trans} transactions)")

def detect_fraud_communities(G):
    """Detect suspicious communities using simple heuristics"""
    print(f"\n🔍 Community Analysis:")
    
    # Find users with multiple fraudulent transactions
    high_risk_users = []
    for user in [n for n in G.nodes() if G.nodes[n]['node_type'] == 'user']:
        fraud_count = sum(1 for _, _, d in G.out_edges(user, data=True) if d['is_fraud'] == 1)
        if fraud_count >= 2:
            high_risk_users.append((user, fraud_count))
    
    high_risk_users.sort(key=lambda x: x[1], reverse=True)
    print(f"   High-risk users (2+ fraud transactions): {len(high_risk_users)}")
    
    # Check for shared merchants among high-risk users
    if len(high_risk_users) >= 2:
        risk_user_merchants = {}
        for user, _ in high_risk_users:
            merchants = set(v for _, v in G.out_edges(user))
            risk_user_merchants[user] = merchants
        
        # Find common merchants
        common_merchants = set.intersection(*risk_user_merchants.values())
        if common_merchants:
            print(f"   Shared merchants among high-risk users: {len(common_merchants)}")
            print(f"   Suspicious merchant network detected! 🚨")

def main():
    """Run graph-based fraud detection example"""
    # Create sample graph
    G = create_sample_transaction_graph()
    
    # Analyze patterns
    analyze_fraud_patterns(G)
    
    # Detect communities
    detect_fraud_communities(G)
    
    print(f"\nGraph analysis completed!")
    print("💡 Next: Try with real transaction datasets for deeper insights")

if __name__ == "__main__":
    main()
