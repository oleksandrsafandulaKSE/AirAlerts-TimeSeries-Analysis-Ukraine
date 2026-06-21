import os
import pandas as pd


def inspect_alerts_dataset(csv_path: str, tail_count: int = 100):
    """
    Loads and profiles the end of the air raid alerts dataset to verify its structure,
    checking for spatial granularity, right-censoring, and threat types.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Target CSV file not found at: {csv_path}")

    print("] Loading dataset...")
    # Read the full dataset to get an accurate structural profile
    df = pd.read_csv(csv_path)

    print("\n=== DATASET OVERVIEW ===")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")

    print("\n=== COLUMN TYPES & MISSING VALUES ===")
    # Capture data types and missing value counts
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df)) * 100
    })
    print(info_df)

    print(f"\n=== LAST {tail_count} ROWS SAMPLE ===")
    # Extract the last N records
    tail_df = df.tail(tail_count)

    # Display a selection of columns or the whole dataframe if small
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(tail_df.head(10))  # Showing first 10 of the tail for terminal readability

    print("\n=== DIAGNOSTIC CHECKLIST FOR V2.0 ===")

    # Check 1: Right-Censoring (Active alerts)
    # Look for common column names representing the end of an alert
    end_col = next((c for c in df.columns if 'end' in c.lower() or 'terminate' in c.lower()), None)
    if end_col:
        active_count = df[end_col].isnull().sum()
        print(f"[✓] Found end-time column: '{end_col}'")
        print(f"    -> Right-Censored (Active) Alerts in dataset: {active_count} ({active_count / len(df) * 100:.2f}%)")
    else:
        print("[!] Warning: Could not automatically identify an alert end-time column.")

    # Check 2: Spatial Granularity
    region_cols = [c for c in df.columns if
                   'region' in c.lower() or 'location' in c.lower() or 'hromada' in c.lower() or 'raion' in c.lower()]
    print(f"[✓] Detected spatial columns: {region_cols}")

    return tail_df

# Example Usage:
# replace 'path_to_your_file.csv' with your actual filename
# sample_tail = inspect_alerts_dataset('path_to_your_file.csv')