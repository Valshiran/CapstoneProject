import time
import matplotlib.pyplot as plt

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer
from forecasting_engine import ForecastingEngine
from inventory_logic import InventoryOptimizationEngine

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
    
    # inventory optmization engine
    optimizer = InventoryOptimizationEngine(
        lead_time_days = 3,
        service_factor_z = 1.65,
        holding_cost_per_unit = 0.5,
        retail_price_per_unit = 10.0
    )
    
    # creating the optimized data frame to run calculations on
    final_df = optimizer.calculate_recommendations(test_results)
    
    # calculating financial totals on the model
    total_holding_model = final_df["holding_cost_model"].sum()
    total_lost_model = final_df["lost_revenue_model"].sum()
    
    total_cost_model = total_holding_model + total_lost_model
    
    # calculating financial totals on the baseline
    total_holding_baseline = final_df["holding_cost_baseline"].sum()
    total_lost_baseline = final_df["lost_revenue_baseline"].sum()
    
    total_cost_baseline = total_holding_baseline + total_lost_baseline
    
    cost_reduction = ((total_cost_baseline - total_cost_model) / total_cost_baseline) * 100
    

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
    print("-" * 60)
    print("FINANCIAL EVALUATION METRICS:")
    print(f"  Model Holding Cost : ${total_holding_model:,.2f}")
    print(f"  Model Lost Revenue : ${total_lost_model:,.2f}")
    print(f"  Model Total Cost   : ${total_cost_model:,.2f}")
    print(f"  Baseline Total Cost: ${total_cost_baseline:,.2f}")
    print(f"  Total Cost Reduction: {cost_reduction:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()