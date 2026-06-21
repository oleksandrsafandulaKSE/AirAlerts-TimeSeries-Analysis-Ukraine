import pandas as pd
import geopandas as gpd
from collections import defaultdict


def build_optimized_features(df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    """
    Engineers Spatiotemporal features (Admin 1 level) using an O(N log N)
    event-driven sweep-line algorithm.
    """
    print("] Calculating Spatial Adjacency Matrix (Strictly Touches)...")
    gdf = gpd.read_file(geojson_path)
    pcode_col = 'adm1_pcode'

    # 1. Clean Spatial Topology (Requirement 2)
    adjacency_dict = {}
    for _, row in gdf.iterrows():
        # .touches() explicitly finds borders without including the region itself
        neighbors = gdf[gdf.geometry.touches(row.geometry)]
        adjacency_dict[row[pcode_col]] = neighbors[pcode_col].tolist()

    print("] Calculating Recency...")
    # Sort chronologically by location
    df = df.sort_values(by=['location_uid', 'started_at']).reset_index(drop=True)
    df['prev_finished_at'] = df.groupby('location_uid')['finished_at'].shift(1)
    df['recency_minutes'] = (df['started_at'] - df['prev_finished_at']).dt.total_seconds() / 60.0
    df['recency_minutes'] = df['recency_minutes'].fillna(-1.0)
    df.drop(columns=['prev_finished_at'], inplace=True)

    print("] Constructing Event-Driven Timeline...")
    # 2. Flatten into a unified event stream (Requirement 1)
    # Start events
    starts = df[['alert_id', 'location_uid', 'started_at']].copy()
    starts.rename(columns={'started_at': 'timestamp'}, inplace=True)
    starts['event_type'] = 1  # Alert started

    # End events
    ends = df[['alert_id', 'location_uid', 'finished_at']].copy()
    ends.rename(columns={'finished_at': 'timestamp'}, inplace=True)
    ends['event_type'] = -1  # Alert ended

    # Combine and sort. If a start and end happen at the exact same millisecond,
    # process the end (-1) before the start (1) to keep the active state accurate.
    events = pd.concat([starts, ends], ignore_index=True)
    events = events.sort_values(by=['timestamp', 'event_type']).reset_index(drop=True)

    print("] Executing Single-Pass Sweep-Line Pass...")
    # 3. Dynamic State Tracking
    active_counts = defaultdict(int)  # Tracks how many active alerts exist per location
    state_records = {}  # Stores the calculated states for each alert_id

    # The O(N) single pass
    for _, event in events.iterrows():
        loc = event['location_uid']

        if event['event_type'] == 1:
            # Calculate the state of the system EXACTLY BEFORE this alert is added
            # Countrywide state: Count locations where active_counts > 0
            cw_active = sum(1 for count in active_counts.values() if count > 0)

            # Neighbor state: Check the active status of strictly adjacent regions
            neighbors = adjacency_dict.get(loc, [])
            nb_active = sum(1 for n in neighbors if active_counts.get(n, 0) > 0)

            # Store the state for this specific alert
            state_records[event['alert_id']] = {
                'countrywide_active': cw_active,
                'neighbor_active_count': nb_active
            }

            # Add this alert to the active system state
            active_counts[loc] += 1

        elif event['event_type'] == -1:
            # Remove this alert from the active system state
            active_counts[loc] -= 1
            # Prevent dictionary bloat/negative counts just in case of data anomalies
            if active_counts[loc] <= 0:
                del active_counts[loc]

    print("] Merging States back to Main Timeline...")
    # Convert state records to a DataFrame and merge on alert_id
    state_df = pd.DataFrame.from_dict(state_records, orient='index').reset_index()
    state_df.rename(columns={'index': 'alert_id'}, inplace=True)

    # Merge back to the original dataframe
    final_df = df.merge(state_df, on='alert_id', how='left')

    # Final chronological sort for modeling
    final_df = final_df.sort_values(by='started_at').reset_index(drop=True)

    print("] Optimized Feature Engineering Complete!")
    return final_df