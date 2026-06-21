import pandas as pd
import geopandas as gpd
from thefuzz import process, fuzz


def clean_region_name(name: str) -> str:
    """
    A lightweight cleaning function (Approach #2) to strip standard
    administrative prefixes/suffixes, giving the fuzzy matcher a cleaner target.
    """
    if pd.isna(name):
        return name

    name = str(name)
    # Order matters: replace suffixes/prefixes with their associated spaces
    name = name.replace(' область', '')
    name = name.replace(' район', '')
    name = name.replace('м. ', '')
    return name.strip()


def map_alerts_to_geojson(processed_df: pd.DataFrame, geojson_path: str, admin_level: int = 1):
    """
    Matches Cyrillic region names to official UN GeoJSON polygons
    using hybrid string cleaning and subset-ratio fuzzy matching.
    """
    print(f"] Loading UN GeoJSON map for Admin Level {admin_level}...")
    gdf = gpd.read_file(geojson_path)

    # 1. Dynamically find the P-Code column and all Name columns
    pcode_col = f'adm{admin_level}_pcode'
    name_cols = [c for c in gdf.columns if f'adm{admin_level}_name' in c]

    if pcode_col not in gdf.columns:
        raise ValueError(f"Critical error: {pcode_col} not found in GeoJSON.")

    # 2. Build a lookup dictionary
    name_to_pcode = {}
    for _, row in gdf.iterrows():
        pcode = row[pcode_col]
        for col in name_cols:
            val = row[col]
            if pd.notna(val):
                name_to_pcode[str(val)] = pcode

    official_names = list(name_to_pcode.keys())
    print("] Matching Cyrillic dataset locations to official Map polygons...")

    match_cache = {}

    def get_best_match(alert_region_name):
        if pd.isna(alert_region_name):
            return None
        if alert_region_name in match_cache:
            return match_cache[alert_region_name]

        # --- APPLY APPROACH #2 (Clean the string) ---
        cleaned_name = clean_region_name(alert_region_name)

        # --- APPLY APPROACH #3 (Smart Token Set Ratio) ---
        best_match, score = process.extractOne(
            cleaned_name,
            official_names,
            scorer=fuzz.token_set_ratio
        )

        # We can safely keep the threshold at 85 because token_set_ratio
        # is highly confident when finding word intersections.
        if score >= 85:
            pcode = name_to_pcode[best_match]
            match_cache[alert_region_name] = pcode
            return pcode
        else:
            print(
                f"  [!] Low match score ({score}) for: '{alert_region_name}' -> Cleaned '{cleaned_name}'. Best guess was '{best_match}'.")
            match_cache[alert_region_name] = None
            return None

    # Apply the matching function
    subset_df = processed_df.copy()
    subset_df['location_uid'] = subset_df['region_name'].apply(get_best_match)

    success_rate = (subset_df['location_uid'].notnull().sum() / len(subset_df)) * 100
    print(f"\n] Geospatial Mapping Complete! Match rate: {success_rate:.2f}%")

    return subset_df

# To execute:
# geo_mapped_df = map_alerts_to_geojson(oblast_alerts, 'ukr_admin_boundaries.geojson/ukr_admin1.geojson', admin_level=1)
# print(geo_mapped_df.head())