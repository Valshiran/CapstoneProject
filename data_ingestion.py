import os
import pandas as pd

class DataIngestion:
    
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_and_validate(self) -> pd.DataFrame:
        """
       Loads dataset, parses dates, and validates data completeness.
        """
        
        # initial message when the method is called to show it is running and looking for the file.
        # will show an error if the file is not found
        print(f"Loading dataset from {self.file_path} and validating raw dataset")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            ) 
        
        # loading the data
        df = pd.read_csv(self.file_path)
        
        # converting date to date/time format
        df["date"] = pd.to_datetime(df["date"])  
        
        # checking data integrity
        assert df["sales"].isnull().sum() == 0
        assert (df["sales"] < 0).sum() == 0
        
        # sorting the data
        df = df.sort_values(by=["store", "item", "date"]).reset_index(drop=True)
        
        # success message
        print(f"Successfully loaded {len(df):,} rows.")
        
        return df
    
    # Add this at the very bottom of data_ingestion.py
if __name__ == "__main__":
    # Test execution with your Kaggle dataset path
    data_path = "data/sales_data.csv"
    ingestion = DataIngestion(data_path)
    df = ingestion.load_and_validate()
    print("Sample output:")
    print(df.head())