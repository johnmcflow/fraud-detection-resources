"""
Interactive Quick Start for Fraud Detection Resources
Run this to get started immediately!
"""
import os
import subprocess
import sys

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'imbalanced-learn']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_packages(packages):
    """Install missing packages"""
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def main():
    """Interactive quick start"""
    print("🛡️  FRAUD DETECTION RESOURCES - QUICK START")
    print("=" * 50)
    print()
    
    print("👋 Welcome to the most comprehensive fraud detection learning platform!")
    print()
    
    # Check environment
    missing = check_requirements()
    if missing:
        print(f"⚠️  Missing packages detected: {', '.join(missing)}")
        install_choice = input("Install automatically? (y/n): ").lower().strip()
        if install_choice == 'y':
            install_packages(missing)
            print("✅ Packages installed successfully!")
        else:
            print("Please install manually: pip install " + " ".join(missing))
            return
    else:
        print("✅ All required packages found!")
    
    print()
    print("🎯 What would you like to try first?")
    print("1. Run basic fraud detector")
    print("2. Try graph-based detection")
    print("3. Explore datasets")
    print("4. View learning roadmap")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("🚀 Running basic fraud detector...")
        os.chdir("code/examples")
        exec(open("basic_fraud_detector.py").read())
    
    elif choice == "2":
        print("🕸️  Running graph-based detection...")
        os.chdir("code/examples")
        exec(open("graph_fraud_basic.py").read())
    
    elif choice == "3":
        print("📊 Available datasets:")
        print("- IEEE-CIS Fraud Detection: https://www.kaggle.com/c/ieee-fraud-detection")
        print("- Credit Card Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("- Elliptic Bitcoin: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        print("- PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1")
    
    elif choice == "4":
        print("📚 52-Week Learning Roadmap:")
        print("Week 1-4: Foundations (Math + Basic ML)")
        print("Week 5-12: Intermediate (Graph Analytics)")  
        print("Week 13-24: Advanced (Deep Learning + GNNs)")
        print("Week 25-52: Expert (Research + Production)")
        print("\nSee full roadmap in README.md!")
    
    else:
        print("Invalid choice. Please run again and select 1-4.")
    
    print()
    print("🎉 Thanks for using Fraud Detection Resources!")
    print("⭐ Star this repo if it helped you: https://github.com/johnmcflow/fraud-detection-resources")
    print("🤝 Contribute: Read CONTRIBUTING.md")

if __name__ == "__main__":
    main()
