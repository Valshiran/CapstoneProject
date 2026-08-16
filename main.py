import time
import matplotlib.pyplot as plt

from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer

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

    # temporary verification of the output, can be removed later
    # this is to make sure the scripts work
    execution_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("               PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"Execution Time       : {execution_time:.2f} seconds")
    print(f"Raw Input Rows       : {len(raw_df):,}")
    print(f"Featured Output Rows : {len(featured_df):,}")
    print(f"Total Columns        : {len(featured_df.columns)}")
    print("-" * 60)
    print("Engineered Columns Created:")
    print(list(featured_df.columns))
    print("-" * 60)
    print("Sample Output Preview:")
    print(
        featured_df[
            [
                "date",
                "store",
                "item",
                "sales",
                "lag_1",
                "lag_7",
                "rolling_mean_7",
                "rolling_std_7",
            ]
        ].head()
    )
    print("=" * 60)


if __name__ == "__main__":
    main()