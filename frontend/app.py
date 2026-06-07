import streamlit as st
import pandas as pd
import requests

from components.visuals import plot_anomaly_3d

API_URL = "http://127.0.0.1:8000/predict-csv"

st.set_page_config(page_title="Financial Anomaly Engine", page_icon="🛡️", layout="wide")

st.title("🛡️ Financial Anomaly Engine")
st.markdown("Upload your transaction logs to instantly identify fraudulent patterns using Unsupervised Machine Learning.")

st.divider()

uploaded_file = st.file_uploader("Upload Transaction Data (CSV format)", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Transmitting to backend AI engine for analysis..."):
        
        files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}

        try:
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                results = response.json()
                st.success("Analysis Complete!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Total Transactions Analyzed", value=f"{results['total_transactions_analyzed']:,}")
                with col2:
                    st.metric(label="Critical Anomalies Flagged", value=f"{results['anomalies_flagged']:,}")
                
                st.divider()

                if results["anomalies_flagged"] > 0:
                    st.subheader(" Detected Anomalies")
                    st.caption("The following transactions exhibit highly unusual patterns compared to the dataset baseline.")

                    anomalies_df = pd.DataFrame(results["anomaly_details"])
                    
                    st.plotly_chart(plot_anomaly_3d(anomalies_df), use_container_width=True)

                    cols = ['is_anomaly', 'anomaly_score', 'amount', 'oldbalanceOrg', 'newbalanceOrg']
                    
                    existing_cols = [c for c in cols if c in anomalies_df.columns] + [c for c in anomalies_df.columns if c not in cols]

                    st.dataframe(anomalies_df[existing_cols], use_container_width=True)
                
            else:
                st.error(f"Backend API Error {response.status_code}: {response.text}")
            
        except requests.exceptions.ConnectionError:
            st.error(" Connection Refused. Is your FastAPI backend running on port 8000?")