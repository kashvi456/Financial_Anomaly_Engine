from fastapi import FastAPI, File, HTTPException, UploadFile
import pandas as pd
import io

from app.models.isolation_forest import AnomalyDetector
from app.services.data_processor import DataProcessor

app = FastAPI(
    title="Financial Anomaly Engine API",
    description="Unsupervised fraud detection using Isolation Forest"
)

detector = AnomalyDetector(contamination=0.05)

@app.get("/")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "Active", "message": "API is running. Send a CSV to /predict-csv"}

@app.post("/predict-csv")
async def predict_anomalies(file: UploadFile = File(...)):
    """
    Accepts a CSV file of financial transactions, processes it,
    and returns the identified anomalies.
    """

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are accepted.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        processed_df = DataProcessor.preprocess_transactions(df)

        results = detector.predict(processed_df)

        df['is_anomaly'] = results['is_anomaly']
        df['anomaly_score'] = results['anomaly_scores']

        anomalies_df = df[df['is_anomaly'] == True]

        return {
            "message": "Data processed successfully.",
            "total_transactions_analyzed": len(df),
            "anomalies_flagged": len(anomalies_df),
            "anomaly_details": anomalies_df.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during Processing:{str(e)}")