import pandas as pd
import os
from src import DataPreprocessor, ModelTrainer, ModelPredictor, ResultVisualizer

class MLClassificationPipeline:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.trainer = ModelTrainer()
        self.predictor = ModelPredictor()
        self.visualizer = ResultVisualizer()
        self.feature_names = None
    
    def train(self, train_path, save_models=True):
        """Train models using data from specified path"""
        print("Loading training data...")
        train_df = pd.read_csv(train_path)
        X_train_raw = train_df.drop(['CLASS', 'ID'], axis=1, errors='ignore')
        y_train = train_df['CLASS']
        
        # Store feature names
        self.feature_names = X_train_raw.columns.tolist()
        
        # Preprocess training data
        print("\nPreprocessing training data...")
        X_train_processed = self.preprocessor.preprocess_data(X_train_raw, fit_transform=True)
        
        # Train and evaluate models
        print("\nTraining models...")
        self.trainer.train(X_train_processed, y_train)
        
        # Save models and preprocessor
        if save_models:
            print("\nSaving models and preprocessor...")
            self.trainer.save_models()
            self._save_preprocessor()
        
        return self.trainer.best_models
    
    def predict(self, test_path, load_models=True):
        """Generate predictions for test data"""
        print("\nProcessing test data...")
        test_df = pd.read_csv(test_path)
        X_test_raw = test_df.drop(['ID', 'CLASS'], axis=1, errors='ignore')
        test_ids = test_df['ID'] if 'ID' in test_df.columns else range(len(test_df))
        
        # Load preprocessor and models if needed
        if load_models:
            print("Loading preprocessor and models...")
            self._load_preprocessor()
            self.predictor.load_models()
        
        # Preprocess test data
        X_test_processed = self.preprocessor.preprocess_data(X_test_raw, fit_transform=False)
        
        # Generate predictions
        predictions, probabilities = self.predictor.predict(X_test_processed, 'test', test_ids)
        
        # If true labels are available, evaluate and visualize results
        if 'CLASS' in test_df.columns:
            y_test = test_df['CLASS']
            self._evaluate_predictions(y_test, predictions, probabilities)
        
        return predictions, probabilities
    
    def _evaluate_predictions(self, y_test, predictions, probabilities):
        """Evaluate and visualize predictions"""
        print("\nTest Set Metrics:")
        
        for model_name in predictions:
            test_results = self.trainer._calculate_metrics(
                y_test, 
                predictions[model_name],
                probabilities[model_name]
            )
            print(f"\n{model_name}:")
            for metric, value in test_results.items():
                print(f"{metric}: {value:.4f}")
            
            # Visualize results for each model
            self.visualizer.plot_confusion_matrices(
                y_test, 
                {model_name: predictions[model_name]}, 
                model_name
            )
            self.visualizer.plot_roc_curves(
                y_test, 
                {model_name: probabilities[model_name]}, 
                model_name
            )
    
    def _save_preprocessor(self):
        """Save preprocessor to file"""
        import joblib
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.preprocessor, 'models/preprocessor.joblib')
        print("Preprocessor saved to models/preprocessor.joblib")
    
    def _load_preprocessor(self):
        """Load preprocessor from file"""
        import joblib
        self.preprocessor = joblib.load('models/preprocessor.joblib')
        print("Preprocessor loaded")