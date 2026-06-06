import pandas as pd
import os
from app.models.isolation_forest import AnomalyDetector
from app.services.data_processor import DataProcessor

def train_and_save_model():
    print("Loading sample data...")
    data_path = "../data/sample_transactions.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}. Please check the path.")
        return
    
    df = pd.read_csv(data_path)

    print("Preprocessing data...")
    processed_df = DataProcessor.preprocess_transactions(df)

    print("Training Isolation Forest Model...")
    detector = AnomalyDetector(contamination=0.05)
    detector.train(processed_df)

    print("Saving model to .pkl...")
    os.makedirs("app/models/saved_models", exist_ok=True)
    model_path = "app/models/saved_models/isolation_forest.pkl"
    detector.save_model(model_path)

    print(f"Success! Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()