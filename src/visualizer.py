import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,roc_curve, auc
import os

class ResultVisualizer:
    def __init__(self):
        self.output_dir = 'visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set seaborn style
        sns.set_style("whitegrid")
        # Set color palette
        sns.set_palette("husl")
        # Set plotting context
        plt.style.use('default')
        sns.set_context("notebook", font_scale=1.2)

    
    def plot_cross_validation_results(self, results):
        """Plot cross-validation results for all models"""
        metrics = list(next(iter(results.values())).keys())
        
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            metric_values = {model: values[metric] 
                           for model, values in results.items()}
            
            # Create bar plot
            bars = plt.bar(range(len(metric_values)), 
                         list(metric_values.values()))
            plt.xticks(range(len(metric_values)), 
                      list(metric_values.keys()), 
                      rotation=45)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')
            
            plt.title(f'Cross-validation {metric} by Model')
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'cv_{metric}.png'))
            plt.close()
    
    def plot_confusion_matrices(self, y_true, predictions_dict, dataset_name):
        """Plot confusion matrix for a model"""
        for model_name, y_pred in predictions_dict.items():
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'{model_name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            
            # Save to file
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, f'cm_{model_name}_{dataset_name}.png'))
            plt.close()
    
    def plot_roc_curves(self, y_true, proba_dict, dataset_name):
        """Plot ROC curve for a model"""
        for model_name, y_proba in proba_dict.items():
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{model_name} ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            
            # Save to file
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(os.path.join(self.output_dir, f'roc_{model_name}_{dataset_name}.png'))
            plt.close()
    
    def plot_feature_importance(self, models_dict, feature_names):
        """Plot feature importance for tree-based models"""
        tree_based_models = {
            name: model for name, model in models_dict.items() 
            if hasattr(model, 'feature_importances_')
        }
        
        for model_name, model in tree_based_models.items():
            plt.figure(figsize=(12, 6))
            importances = pd.DataFrame({
                'features': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=importances.head(20), 
                       x='importance', y='features')
            plt.title(f'Top 20 Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 
                                    f'feature_importance_{model_name}.png'))
            plt.close()