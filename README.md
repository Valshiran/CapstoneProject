## Configuration Parameters (`config.json`)

Operational parameters, model split thresholds, and visualization targets can be adjusted in `config.json` without modifying codebase logic:

### Prescriptive Inventory Settings (`inventory_settings`)
* **`lead_time_days`** *(int)*: Supplier delivery lead time in days (e.g., `3`). Used to scale baseline safety stock and calculate reorder thresholds.
* **`service_factor_z`** *(float)*: Z-score multiplier corresponding to the desired cycle service level (e.g., `1.65` for ~95% in-stock availability).
* **`holding_cost_per_unit`** *(float)*: Estimated daily cost ($) incurred per unit of excess inventory stored.
* **`retail_price_per_unit`** *(float)*: Retail value ($) lost per unit during a stockout event.

### Forecasting Settings (`forecasting_settings`)
* **`test_start_year`** *(int)*: The holdout test split boundary year (e.g., `2017`). Data prior to this year is used for training, while this year is reserved for model evaluation.
 
 NOTE: It is not recommended to change the test_start_year for this specific dataset.

### Visual Plotting Settings (`visuals`)
* **`target_store`** *(int)*: Store ID selected for generating the quarterly evaluation time-series plots.
* **`target_item`** *(int)*: Item ID selected for generating the quarterly evaluation time-series plots.


### ROOT/FOLDER STRUCTURE

├── data/
│   └── sales_data.csv               # Input dataset (2013–2017 historical sales)
├── outputs/
│   ├── inventory_recommendations.csv # Deliverable: Output recommendation log
│   ├── model_performance_summary_Q1.png # Deliverable: Visual check (Q1)
│   ├── model_performance_summary_Q2.png # Deliverable: Visual check (Q2)
│   ├── model_performance_summary_Q3.png # Deliverable: Visual check (Q3)
│   └── model_performance_summary_Q4.png # Deliverable: Visual check (Q4)
├── data_ingestion.py                # Data loading & validation engine
├── feature_engineering.py           # Time-series lag & rolling feature generation
├── forecasting_engine.py            # Chronological split & Ridge regression model
├── inventory_logic.py                # Prescriptive safety stock & reorder formulas
├── main.py                          # End-to-end pipeline orchestrator
├── .gitignore                       # Excludes cache and workspace configs
└── README.md                        # Project documentation