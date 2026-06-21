import pandas as pd
import uuid


def process_alert_data(csv_path: str) -> pd.DataFrame:
    """
    Transforms the raw air raid alerts CSV into the Version 2.0 schema,
    preparing it for spatiotemporal feature engineering.
    """
    print("] Loading raw data...")
    df = pd.read_csv(csv_path)

    print("] Parsing datetimes to UTC...")
    # Convert strings to timezone-aware UTC datetime objects
    df['started_at'] = pd.to_datetime(df['started_at'], utc=True)
    df['finished_at'] = pd.to_datetime(df['finished_at'], utc=True)

    print("] Calculating alert durations...")
    # Calculate duration in minutes.
    # (If finished_at was NaT/Null, duration would naturally be NaN)
    df['duration_minutes'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 60.0

    # Flag active alerts (none in this specific historical batch, but good for V2.0 pipeline)
    df['is_active'] = df['finished_at'].isnull()

    print("] Standardizing spatial hierarchy...")
    # Create a unified 'region_name' based on the alert 'level'
    # If level is 'oblast', use oblast name. If 'raion', use raion name, etc.
    df['region_name'] = df.apply(
        lambda row: row['hromada'] if row['level'] == 'hromada'
        else (row['raion'] if row['level'] == 'raion' else row['oblast']),
        axis=1
    )

    # Rename for clarity to match V2.0 schema
    df.rename(columns={'oblast': 'parent_oblast', 'level': 'location_type'}, inplace=True)

    print("] Generating Unique IDs and Placeholders...")
    # Generate a unique ID for each alert event
    df['alert_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Placeholders for future data enrichment
    df['threat_type'] = 'Unknown'
    df['incident_notes'] = None

    # Sort chronologically
    df = df.sort_values(by='started_at').reset_index(drop=True)

    # Reorder columns to match our proposed V2.0 Schema
    final_cols = [
        'alert_id', 'region_name', 'parent_oblast', 'location_type',
        'threat_type', 'started_at', 'finished_at', 'duration_minutes',
        'is_active', 'incident_notes'
    ]

    clean_df = df[final_cols]
    print("] ETL Complete!")

    return clean_df

# To execute:
# processed_df = process_alert_data('path_to_your_dataset.csv')
# print(processed_df.head())