import pandas as pd
import joblib
import os

class ModelPredictor:
    def __init__(self):
        self.models = {}
        
    def load_models(self, models_dir='models'):
        """Load all saved models"""
        print("Loading models...")
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.joblib') and model_file != 'preprocessor.joblib':
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(models_dir, model_file)
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name}")
    
    def predict(self, X, dataset_name, ids=None):
        """Generate predictions from all models"""
        if ids is None or ids.empty:
            ids = pd.Series(range(len(X)))
            
        predictions_df = pd.DataFrame({'ID': ids})
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)
            predictions[model_name] = y_pred
            probabilities[model_name] = y_proba[:, 1]  # Probability of positive class
            
            # Save probabilities to dataframe
            for class_idx in range(y_proba.shape[1]):
                col_name = f'{model_name}_Class_{class_idx}_Prob'
                predictions_df[col_name] = y_proba[:, class_idx]
        
        # Save predictions
        filename = f'predictions_{dataset_name}.csv'
        predictions_df.to_csv(filename, index=False)
        print(f"Predictions saved to {filename}")
        
        return predictions, probabilities