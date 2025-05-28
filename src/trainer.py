import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix
from src import get_models

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.results = {}
        
    def train(self, X, y, cv_folds=5):
        """Main training pipeline"""
        self.models = get_models()  # Get model definitions
        self._cross_validate(X, y, cv_folds)
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
    
    def _validate_model(self, model_name, model, X, y, skf):
        fold_results = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            metrics = self._train_and_evaluate_fold(
                model, X, y, train_idx, val_idx, fold)
            fold_results.append(metrics)
        self.results[model_name] = fold_results
    
    def _train_and_evaluate_fold(self, model, X, y, train_idx, val_idx, fold):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = self._calculate_metrics(y_val, y_pred, y_proba)
        print(f"  Fold {fold+1}: " + ", ".join(
            [f"{k}: {v:.4f}" for k, v in metrics.items()]))
        return metrics
    
    def _train_final_models(self, X, y):
        """Train final models on full training data"""
        print("\nTraining final models...")
        for model_name, model in self.models.items():
            print(f"Training final {model_name}...")
            model.fit(X, y)
            self.best_models[model_name] = model
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        for model_name, model in self.best_models.items():
            path = os.path.join(output_dir, f'{model_name}.joblib')
            joblib.dump(model, path)
            print(f"Saved {model_name} to {path}")
    
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