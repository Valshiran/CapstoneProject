import numpy as np
import pandas as pd

class InventoryOptimizationEngine:
    
    def __init__(
        self,
        # default values if someone removes values from the config file
        lead_time_days: int = 3,
        service_factor_z: float = 1.65,
        holding_cost_per_unit: float = 0.5,
        retail_price_per_unit: float = 10.0
    ):
        """
        Initializes the InventoryOptimizationEngine with parameters for inventory calculations.
        """
        self.lead_time_days = lead_time_days
        self.service_factor_z = service_factor_z
        self.holding_cost_per_unit = holding_cost_per_unit
        self.retail_price_per_unit = retail_price_per_unit
        
    def calculate_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates safety stock, dynamic reorder points, and financial costs.

        Computes values for both the machine learning model and the naive
        baseline.
        """
        
        # status message to show what portion of the pipeline/flow it is working on
        print("Calculating inventory parameters and cost evaluations...")
        
        # creates a copy of the df, so a clean copy of the raw df is still in memory
        df = df.copy()
        
        # =========================================================
        # DYNAMIC SAFETY STOCK & REORDER POINT FORMULAS
        # Safety Stock = Z * std_dev * sqrt(lead_time)
        # Reorder Point = Predicted Lead Time Demand + Safety Stock
        # =========================================================
        
        lead_time_mult = np.sqrt(self.lead_time_days)
        
        # model prescriptions
        df["safety_stock"] = (self.service_factor_z * df["rolling_std_7"] * lead_time_mult)
        df["reorder_point"] = (df["pred_demand"] * self.lead_time_days) + df["safety_stock"]
        
        # baseline prescriptions
        df["safety_stock_baseline"] = (self.service_factor_z * df["rolling_std_7"] * lead_time_mult)  
        df["reorder_point_baseline"] = (self.service_factor_z * df["rolling_std_7"]* lead_time_mult)
        
        # ======================================================== 
        # FINANCIAL COST MODELING
        # Holding Cost - when forecast > actual sales (Overstock)
        # Lost Revenue - when actual sales > forecast (Stockout)
        # ========================================================
        
        # model financials
        model_error = df["pred_demand"] - df["sales"]
        df["holding_cost_model"] = np.where(model_error > 0, model_error * self.holding_cost_per_unit, 0.0)
        df["lost_revenue_model"] = np.where(model_error < 0, np.abs(model_error) * self.retail_price_per_unit, 0.0)
        
        # baseline financials
        baseline_error = df["baseline_pred"] - df["sales"]
        df["holding_cost_baseline"] = np.where(baseline_error > 0, baseline_error * self.holding_cost_per_unit, 0.0)
        df["lost_revenue_baseline"] = np.where(baseline_error < 0, np.abs(baseline_error) * self.retail_price_per_unit, 0.0)
        
        # status message to show where we are in the process 
        print("Inventory optimization calculations complete")
        
        return df