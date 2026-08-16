import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import Ridge


class ForecastingEngine:

    def __init__(self, test_start_year: int = 2017):
        self.test_start_year = test_start_year
        self.model = Ridge(alpha=1.0)

    def train_and_predict(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict]:
        """
        Splits data chronologically, fits Ridge model, and computes evaluation metrics.
        """
        print(
            f"Training forecasting model (Holdout Test Year: {self.test_start_year})..."
        )

        # Chronological Split (Prevents future data leak into past training)
        train_df = df[df["year"] < self.test_start_year].copy()
        test_df = df[df["year"] >= self.test_start_year].copy()

        # define feature columns (excludes metadata, target, and baseline columns)
        # these columns are created in feature_engineering.py
        feature_cols = [
            "year",
            "month",
            "day",
            "dayofweek",
            "dayofyear",
            "is_weekend",
            "lag_1",
            "lag_7",
            "lag_30",
            "lag_365",
            "rolling_mean_7",
            "rolling_std_7",
            "rolling_mean_30",
        ]

        X_train, y_train = train_df[feature_cols], train_df["sales"]
        X_test, y_test = test_df[feature_cols], test_df["sales"]

        # fit the Machine Learning Model
        self.model.fit(X_train, y_train)

        # generate Predictions & Naive Baseline (Same Day Last Week)
        test_df["pred_demand"] = self.model.predict(X_test)
        test_df["baseline_pred"] = test_df["lag_7"]

        # compute Regression Metrics
        mae_model = mean_absolute_error(y_test, test_df["pred_demand"])
        rmse_model = root_mean_squared_error(y_test, test_df["pred_demand"])

        mae_base = mean_absolute_error(y_test, test_df["baseline_pred"])
        rmse_base = root_mean_squared_error(y_test, test_df["baseline_pred"])

        metrics = {
            "MAE_Model": mae_model,
            "RMSE_Model": rmse_model,
            "MAE_Baseline": mae_base,
            "RMSE_Baseline": rmse_base
        }

        # success message with the results of the model
        print(
            f"Forecasting complete. Model MAE: {mae_model:.2f} | Baseline MAE: {mae_base:.2f}"
        )

        return test_df, metrics