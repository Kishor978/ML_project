import pandas as pd
import joblib
import os

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.thresholds = {}
        
    def load_models(self, models_dir='models'):
        """Load all saved models"""
        print("Loading models...")
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.pkl'):
                if model_file == 'optimal_thresholds.pkl':
                    self.thresholds = joblib.load(os.path.join(models_dir, model_file))
                    print("Loaded optimal thresholds")
                elif model_file != 'preprocessor.pkl':
                    model_name = os.path.splitext(model_file)[0]
                    model_path = os.path.join(models_dir, model_file)
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name}")
    
    def predict(self, X, dataset_name, ids=None):
        """Generate predictions from all models with custom thresholds for RF and XGBoost"""
        if ids is None or ids.empty:
            ids = pd.Series(range(len(X)))
            
        predictions_df = pd.DataFrame({'ID': ids})
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            # Get probabilities
            y_proba = model.predict_proba(X)
            probabilities[model_name] = y_proba[:, 1]
            
            # Apply custom threshold for RF and XGBoost if available
            if model_name in self.thresholds:
                threshold = self.thresholds[model_name]
                y_pred = (y_proba[:, 1] >= threshold).astype(int)
                print(f"Using custom threshold {threshold:.4f} for {model_name}")
            else:
                y_pred = model.predict(X)
            
            predictions[model_name] = y_pred
            
            # Save probabilities to dataframe
            for class_idx in range(y_proba.shape[1]):
                col_name = f'{model_name}_Class_{class_idx}_Prob'
                predictions_df[col_name] = y_proba[:, class_idx]
            
            # Also save predictions using custom thresholds
            predictions_df[f'{model_name}_Prediction'] = y_pred
        
        # Save predictions
        filename = f'predictions_{dataset_name}.csv'
        predictions_df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")
        
        return predictions, probabilities