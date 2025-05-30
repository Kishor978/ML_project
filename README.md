# Machine Learning Classification Project
## Overview
This project involves developing a machine learning classification solution using various algorithms. The main goal appears to be binary classification on a dataset with numerous features (3000+), with models evaluated on metrics including accuracy, F1 score, recall, specificity, and AUROC.

Usage
The project appears to provide both notebook-based exploration and script-based execution:
- Notebooks in the **notebook** directory for exploration and experimentation
- **train.py** for training models
- **test.py** for making predictions on test data
- **blind_test.py** for making prediction on blind test data

### Dependencies
The project requires several Python packages, which can be installed from the **requirements.txt** file.
#### To use this project:
- Install dependencies: pip install -r requirements.txt
- Explore the notebooks to understand the data and modeling process
- Run training: python train.py
- Generate predictions: python blind_test.py

### Data
The project uses three main data files:
- train_set.csv: Used for training the models
- test_set.csv: Used for evaluating model performance
- blinded_test_set.csv: Likely used for final predictions
The dataset appears to have numerous features (3000+) and binary targets.

### Preprocessing Steps
From the notebooks, the preprocessing pipeline includes:

- Handling infinite and NaN values
- Feature selection using variance thresholding
- Removing highly correlated features
- Feature clustering using agglomerative clustering
- Dimensionality reduction with PCA
- Data scaling with StandardScaler
- Handling class imbalance with SMOTE resampling

### Models Implemented
Several classification models have been explored:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
- Naive Bayes
### Hyperparameter Tuning
The notebooks show evidence of hyperparameter tuning using:


- GridSearchCV
- Optuna for more efficient parameter search

### Evaluation
Models are evaluated using 5-fold cross-validation with metrics including:

- Accuracy
- F1 Score
- Recall
- Specificity
- AUROC
### Visualization
The project includes a **visualizer.py** module for generating visualizations, with outputs stored in the visualizations directory.