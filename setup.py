from setuptools import setup, find_packages

setup(
    name="fraud-detection-resources",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open('requirements.txt').readlines()
    ],
    author="BSides Querétaro Contributors",
    description="Comprehensive fraud detection learning resources",
    url="https://github.com/johnmcflow/fraud-detection-resources",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
```

### **2. Actualizar .gitignore**:
Agregar estas líneas:
```
# Docker
.docker/
docker-compose.override.yml

# Jupyter
.ipynb_checkpoints/

# Models
models/
*.pkl
*.joblib

# Data
datasets/raw/
datasets/processed/
*.csv
!datasets/sample_data/*.csv
