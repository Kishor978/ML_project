from src import MLClassificationPipeline

def main():
    # Define paths
    train_path = 'dataset/train_set.csv'
    
    # Initialize pipeline and train models
    print("=== Starting Training Pipeline ===")
    pipeline = MLClassificationPipeline()
    pipeline.train(train_path, save_models=True)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()