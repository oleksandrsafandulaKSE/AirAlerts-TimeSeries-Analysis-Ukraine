import os
import pandas as pd
from Investigating_dataset import *
from transformation_data import process_alert_data
from mapping_alerts import map_alerts_to_geojson
from bulding_features import build_optimized_features
from clean_grid import generate_clean_training_grid
from model_training import train_risk_model

# sample_tail = inspect_alerts_dataset('official_data_uk.csv') to check the structure of the dataset

processed_df = process_alert_data('official_data_uk.csv')

oblast_alerts = processed_df[processed_df['location_type'] == 'oblast']
geo_mapped_df = map_alerts_to_geojson(oblast_alerts, 'ukr_admin_boundaries.geojson/ukr_admin1.geojson', admin_level=1)

final_feature_df = build_optimized_features(geo_mapped_df, 'ukr_admin_boundaries.geojson/ukr_admin1.geojson')
print(final_feature_df[['started_at', 'region_name', 'recency_minutes', 'countrywide_active', 'neighbor_active_count']].head(15))


# ---------------------------------------------------------
# END OF PHASE 5 SCRIPT
# ---------------------------------------------------------

print("Starting Phase 5: Generating Spatiotemporal Training Matrix...")

# 1. Execute the final grid generator
training_matrix = generate_clean_training_grid(
    alerts_df=final_feature_df,
    freq='1h'
)

# 2. SAVE THE MATRIX TO DISK (This is Step 3!)
# The index=False parameter is critical so pandas doesn't save a useless column of row numbers.
print("] Saving training matrix to CSV for the dashboard...")
training_matrix.to_csv('training_matrix.csv', index=False)
print("] Saved successfully as 'training_matrix.csv'.")

# 3. Train the production LightGBM model
production_model = train_risk_model(training_matrix, scale_weight=13.65)

# 4. Save the trained model to disk for the dashboard
production_model.booster_.save_model('lgbm_risk_model.txt')
print("] Pipeline Complete. Ready for Phase 6 Deployment.")