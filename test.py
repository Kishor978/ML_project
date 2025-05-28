from src.pipeline import MLClassificationPipeline

def main():
    # Define paths
    test_path = 'dataset/test_set.csv'
    
    # Initialize pipeline and make predictions
    print("=== Starting Prediction Pipeline ===")
    pipeline = MLClassificationPipeline()
    predictions, probabilities = pipeline.predict(test_path, load_models=True)
    print("\nPrediction completed successfully!")

if __name__ == "__main__":
    main()