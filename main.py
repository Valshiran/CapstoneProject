import time
import matplotlib.pyplot as plt

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer
from forecasting_engine import ForecastingEngine

def main():
    start_time = time.time()
    data_path = "data/sales_data.csv"

    # calls the class/method from data_ingestion.py to load the data
    ingestion = DataIngestion(data_path)
    # creating a raw copy of the df
    raw_df = ingestion.load_and_validate()

    # calls the class/method from feature_engineering.py to create features
    fe = FeatureEngineer()
    featured_df = fe.create_features(raw_df)
    
    # calls the class/method from forecasting_engine.p to train/predict
    forecaster = ForecastingEngine(test_start_year=2017)
    test_results, metrics = forecaster.train_and_predict(featured_df)

    # temporary verification of the output, can be removed later
    # this is to make sure the scripts work
    execution_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("               PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"Execution Time       : {execution_time:.2f} seconds")
    print(f"Test Set Row Count   : {len(test_results):,}")
    print("-" * 60)
    print("REGRESSION PERFORMANCE METRICS:")
    print(f"  Model MAE          : {metrics['MAE_Model']:.2f} units")
    print(f"  Baseline MAE       : {metrics['MAE_Baseline']:.2f} units")
    print(f"  Model RMSE         : {metrics['RMSE_Model']:.2f} units")
    print(f"  Baseline RMSE      : {metrics['RMSE_Baseline']:.2f} units")
    print("=" * 60)

if __name__ == "__main__":
    main()