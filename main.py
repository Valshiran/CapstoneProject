import os
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

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
    
    # export the csv file (deliverable)
    print("Exporting deliverables and generating visualizations...")

    # Export Recommendation CSV
    output_cols = [
        "date",
        "store",
        "item",
        "sales",
        "pred_demand",
        "safety_stock",
        "reorder_point",
        "holding_cost_model",
        "lost_revenue_model",
    ]
    
    output_csv_path = os.path.join("outputs", "inventory_recommendations.csv")
    final_df[output_cols].to_csv(output_csv_path, index=False)
    print(f"      -> Exported '{output_csv_path}'")
    
    # calculating financial totals on the model
    total_holding_model = final_df["holding_cost_model"].sum()
    total_lost_model = final_df["lost_revenue_model"].sum()
    
    total_cost_model = total_holding_model + total_lost_model
    
    # calculating financial totals on the baseline
    total_holding_baseline = final_df["holding_cost_baseline"].sum()
    total_lost_baseline = final_df["lost_revenue_baseline"].sum()
    
    total_cost_baseline = total_holding_baseline + total_lost_baseline
    
    # calculating total cost reduction
    cost_reduction = ((total_cost_baseline - total_cost_model) / total_cost_baseline) * 100
    
    # =====================================================
    # Generate and save performance plot
    # =====================================================
    # Filter for representative series (Store 1, Item 1)
    sample_series = final_df[
        (final_df["store"] == 1) & (final_df["item"] == 1)
    ].copy()
    sample_series["date"] = pd.to_datetime(sample_series["date"])
    sample_series["quarter"] = sample_series["date"].dt.quarter

    # Quarter metadata for titles
    q_titles = {
        1: "Q1: Jan - Mar 2017",
        2: "Q2: Apr - Jun 2017",
        3: "Q3: Jul - Sep 2017",
        4: "Q4: Oct - Dec 2017",
    }
    
    # status message to show where we are at in the pipeline/flow
    print("\nGenerating quarterly performance plots...")

    # I initially graphed the time series for the whole year of 2017 on one chart
    # I did not like the look of it (it looked really choppy/busy), so I separated it out into 4 quarters
    # To do this, I felt the easiest way to do this would be using a for loop to generate them, instead of putting the code 4 times

    for q in range(1, 5):
        q_data = sample_series[sample_series["quarter"] == q].copy()

        plt.figure(figsize=(10, 4.5))

        # 1. Actual Sales (Background Context)
        plt.plot(
            q_data["date"],
            q_data["sales"],
            label="Actual Sales",
            alpha=0.5,
            color="grey",
            linewidth=1.2,
        )

        # 2. Predicted Demand (Model Output)
        plt.plot(
            q_data["date"],
            q_data["pred_demand"],
            label="Predicted Demand (Model)",
            color="#1f77b4",
            linestyle="--",
            linewidth=1.5,
        )

        # 3. Dynamic Buffer Threshold (3-Day Rolling Mean for Visual Smoothness)
        smoothed_buffer = (
            q_data["pred_demand"] + q_data["safety_stock"]
        ).rolling(window=3, min_periods=1).mean()
        
        plt.plot(
            q_data["date"],
            smoothed_buffer,
            label="Daily Safety Buffer Threshold",
            color="#d62728",
            alpha=0.85,
            linewidth=1.5,
        )

        # Formatting
        plt.title(
            f"Store 1, Item 1: Demand Forecast & Buffer ({q_titles[q]})",
            fontsize=11,
            pad=10,
        )
        plt.xlabel("Date", fontsize=9)
        plt.ylabel("Units", fontsize=9)
        plt.legend(loc="upper left", fontsize=8)
        plt.grid(True, linestyle=":", alpha=0.6)

        # Format X-Axis to show weeks/months clearly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        plt.tight_layout()

        # Save with Q suffix in my outputs folder
        filename = f"model_performance_summary_Q{q}.png"
        filepath = os.path.join("outputs", f"model_performance_summary_Q{q}.png")
        plt.savefig(filepath, dpi=300)
        plt.close()  # Close plot memory before next iteration
        print(f"      -> Saved plot as '{filename}' in outputs")

    # temporary verification of the output, can be removed later
    # this is to make sure the scripts work
    
    # I ended up leaving these in because I like the look of the summary in the terminal
    # it's a nice visual showing the high level results and confirmation that it's working
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