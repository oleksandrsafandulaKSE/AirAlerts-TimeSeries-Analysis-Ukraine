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

import pandas as pd


def inspect_english_dataset(csv_path: str):
    """
    Inspects the English dataset to evaluate its spatial naming conventions
    and check for potential transliteration or structural mismatches.
    """
    print("] Loading English dataset...")
    df = pd.read_csv(csv_path)

    print("\n=== ENGLISH DATASET SPATIAL COLUMNS ===")
    spatial_cols = [c for c in df.columns if
                    any(x in c.lower() for x in ['oblast', 'raion', 'hromada', 'region', 'location'])]
    print(f"Detected spatial columns: {spatial_cols}")

    # Print data types and null counts for these columns
    print(df[spatial_cols].isnull().sum())

    print("\n=== UNIQUE SAMPLE VALUES FOR ALIGNMENT CHECK ===")
    for col in spatial_cols:
        # Get top 10 unique non-null values to inspect formatting
        unique_samples = df[col].dropna().unique()[:10]
        print(f"\nTop 10 unique values in '{col}':")
        for sample in unique_samples:
            print(f"  - '{sample}'")

    print("\n=== CRITICAL TESTING CRITERIA ===")
    print("When you look at the output, check for these 3 red flags:")
    print("1. Suffixes: Do they write 'Kyiv' or 'Kyiv Oblast'? 'Kharkiv' or 'Kharkivskyi Raion'?")
    print("2. Word Separators: Are multi-word Hromadas separated by spaces, hyphens, or underscores?")
    print("3. Character Encoding: Are there any unintended special characters or apostrophes (like ' or ’)?")

    return df[spatial_cols].head(10)

# Example Usage:
# inspect_english_dataset('path_to_your_english_dataset.csv')