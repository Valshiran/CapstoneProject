import pandas as pd

class FeatureEngineer:

    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal indices, lag features, and rolling statistics.
        """
        # used as a status message to show where the whole process is
        print("Performing time-series feature engineering...")
        
        # creats a copy of the df, so a clean copy of the raw df is still in memory
        df = df.copy()

        # breaking down the dates into multiple features to help the model learn better
        # also meant to help it learn seasonal, monthly, and weekly patterns
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["dayofweek"] = df["date"].dt.dayofweek
        df["dayofyear"] = df["date"].dt.dayofyear
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # Historical lag and rolling window features per store-item pair
        grouped = df.groupby(["store", "item"])["sales"]

        # sales yesterday
        df["lag_1"] = grouped.shift(1)
        # sales same day last week
        df["lag_7"] = grouped.shift(7)
        # sales approximately a month ago (not exactly 1 month since some months have 31 days, and February has 28/29)
        df["lag_30"] = grouped.shift(30)
        # sales same day last year (used to help with seasonality)
        df["lag_365"] = grouped.shift(365)

        # Rolling statistics (shifted by 1 to prevent leakage)
        # for the sake of the project, we are keeping the windows being calculated simple
        # more complex rolling windows can be calculated, but this makes other areas of the project more complex as well
        
        # calculates a 7-day average demand on a rolling basis
        df["rolling_mean_7"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=7).mean()
        )
        # calculates a 7-day standard deviation
        df["rolling_std_7"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=7).std()
        )
        # calculates a 30-day average demand on a rolling basis
        df["rolling_mean_30"] = grouped.transform(
            lambda x: x.shift(1).rolling(window=30).mean()
        )

        # Drop NaNs resulting from lag/rolling calculations
        df = df.dropna().reset_index(drop=True)
        
        # success message
        print(f"Feature engineering complete. Dataset size: {len(df):,} rows.")
        
        return df