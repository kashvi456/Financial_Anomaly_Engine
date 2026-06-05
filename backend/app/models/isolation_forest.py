import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def train(self, X: pd.DataFrame) -> None:
        """
        Trains the Isolation Forest model on the provided numerical feature matrix.
        """
        self.model.fit(X)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> dict:
        """
        Predict anomalies.
        Returns:
            -  is_anomaly: True if flagged as anomaly (-1 from sklearn), False otherwise (1)
            -  anomaly_score: The raw anomaly score (lower means more anomalous)
        """

        if not self.is_trained:
            self.train(X)
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)

        formatted_predictions = [True if pred == -1 else False for pred in predictions]

        return {
            "is_anomaly": formatted_predictions,
            "anomaly_scores": scores.tolist()
        }
    
    def save_model(self, file_path: str) -> None:
        """ Saves the trained model state to a file."""
        joblib.dump(self.model, file_path)

    def load_model(self, file_path: str) -> None:
        """Loads a pre-trained model state from a file."""
        if os.path.exists(file_path):
            self.model = joblib.load(file_path)
            self.is_trained = True
