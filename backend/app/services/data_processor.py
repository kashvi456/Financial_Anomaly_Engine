import pandas as pd

class DataProcessor:
    @staticmethod
    def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and transforms raw Paysim transaction data into
        numerical vectors ready for the Isolation Forest model.
        """
        processed_df = df.copy()

        features = ['amount', 'oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest','newbalanceDest']

        processed_df = processed_df.dropna(subset=features)

        if 'type' in processed_df.columns:
            type_dummies = pd.get_dummies(processed_df['type'],  prefix='type', drop_first=True)
            processed_df = pd.concat([processed_df[features], type_dummies], axis=1)
        else:
            processed_df = processed_df[features]
        
        processed_df = processed_df.astype(float)

        return processed_df

