import pandas as pd
from pathlib import Path

def check_parquet_file(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"Successfully read Parquet file: {file_path}")
        print(df)
        return df
    except Exception as e:
        print(f"Error reading Parquet file: {e}")

if __name__ == "__main__":
    file_path = Path("enhanced/dick_nielsen_filters_audit_enhanced_20260219.parquet")  # Update this path to your Parquet file
    enhanced = check_parquet_file(file_path)

    file_path_multi = Path("enhanced_multi/dick_nielsen_filters_audit_enhanced_20260219.parquet")  # Update this path to your Parquet file
    enhanced_multi = check_parquet_file(file_path_multi)
