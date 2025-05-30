import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from src import get_models
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.optimal_thresholds = {}
        
    def train(self, X, y, cv_folds=5):
        """Main training pipeline"""
        self.models = get_models()  # Get model definitions
        self._cross_validate(X, y, cv_folds)
        self._find_optimal_thresholds(X, y)
        self._train_final_models(X, y)
        
    def _cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation for all models"""
        print("Starting cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.results = {model_name: [] for model_name in self.models}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}")
            self._validate_model(model_name, model, X, y, skf)
            
        return self._summarize_results()
    
    def _validate_model(self, model_name, model, X, y, skf):
        fold_results = []
        use_smote = model_name in ["RandomForest", "XGBoost"]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Apply SMOTE oversampling for RF and XGBoost
            if use_smote:
                smote = SMOTE(random_state=42)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                metrics = self._train_and_evaluate_fold(
                    model, X_train_resampled, y_train_resampled, X_val, y_val, fold)
            else:
                metrics = self._train_and_evaluate_fold(
                    model, X_train, y_train, X_val, y_val, fold)
                
            fold_results.append(metrics)
        self.results[model_name] = fold_results
    
    def _train_and_evaluate_fold(self, model, X_train, y_train, X_val, y_val, fold):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_proba)
        print(f"  Fold {fold+1}: " + ", ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.items()]))
        return metrics
    
    def _find_optimal_thresholds(self, X, y):
        """Find optimal prediction thresholds for XGBoost and RandomForest"""
        print("\nFinding optimal thresholds...")
        
        # Only optimize thresholds for tree-based models
        models_to_optimize = {name: model for name, model in self.models.items() 
                             if name in ["RandomForest", "XGBoost"]}
        
        # Apply SMOTE once for evaluation
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Create validation set with SMOTE data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(X_resampled, y_resampled))
        X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
        y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
        
        for model_name, model in models_to_optimize.items():
            print(f"Optimizing threshold for {model_name}...")
            
            # Train model on training portion
            model.fit(X_train, y_train)
            
            # Get probabilities on validation set
            y_probs = model.predict_proba(X_val)[:, 1]
            
            # Try different thresholds
            thresholds = np.linspace(0.1, 0.9, 50)
            f1_scores = []
            
            for threshold in thresholds:
                y_pred = (y_probs >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred)
                f1_scores.append(f1)
            
            # Find best threshold
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            self.optimal_thresholds[model_name] = best_threshold
            
            print(f"  Optimal threshold: {best_threshold:.4f} (F1: {f1_scores[best_idx]:.4f})")
            
    def _train_final_models(self, X, y):
        """Train final models on full training data"""
        print("\nTraining final models...")
        for model_name, model in self.models.items():
            print(f"Training final {model_name}...")
            
            # Apply SMOTE for RF and XGBoost final models
            if model_name in ["RandomForest", "XGBoost"]:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                model.fit(X_resampled, y_resampled)
            else:
                model.fit(X, y)
                
            self.best_models[model_name] = model
    
    def _summarize_results(self):
        """Summarize cross-validation results with mean and std for each metric"""
        summary = {}
        
        for model_name, fold_results in self.results.items():
            summary[model_name] = {}
            # Convert list of dicts to dict of lists
            metrics_dict = {
                metric: [fold[metric] for fold in fold_results]
                for metric in fold_results[0].keys()
            }
            
            # Calculate mean and std for each metric
            for metric, values in metrics_dict.items():
                mean_value = np.mean(values)
                std_value = np.std(values)
                summary[model_name][metric] = {
                    'mean': mean_value,
                    'std': std_value
                }
                print(f"{model_name} {metric}: {mean_value:.4f} ± {std_value:.4f}")
        
        return summary
    
    @staticmethod
    def _calculate_metrics(y_true, y_pred, y_proba):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'AUROC': roc_auc_score(y_true, y_proba),
            'Sensitivity': recall_score(y_true, y_pred),
            'Specificity': specificity,
            'F1': f1_score(y_true, y_pred)
        }
    
    def save_models(self, output_dir='models'):
        """Save trained models and optimal thresholds"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.best_models.items():
            path = os.path.join(output_dir, f'{model_name}.joblib')
            joblib.dump(model, path)
            print(f"Saved {model_name} to {path}")
        
        # Save thresholds
        if self.optimal_thresholds:
            thresholds_path = os.path.join(output_dir, 'optimal_thresholds.joblib')
            joblib.dump(self.optimal_thresholds, thresholds_path)
            print(f"Saved optimal thresholds to {thresholds_path}")