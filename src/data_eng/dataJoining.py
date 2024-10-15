import pandas as pd
import geopandas as gpd

# Apartment Listing Data
# ------------------------------------------------------------------------------------
# Load the apartment rental data from the CSV file
df_apartments = pd.read_csv(
    "data_source/apartments_for_rent_classified_100K.csv", 
    sep=";", encoding='cp1252', low_memory=False
)
# Drop rows with missing latitude or longitude values
df_apartments = df_apartments.dropna(subset=['latitude', 'longitude'])

# Create a GeoDataFrame from latitude and longitude columns
gdf_apartments = gpd.GeoDataFrame(
    df_apartments, 
    geometry=gpd.points_from_xy(df_apartments.longitude, df_apartments.latitude), 
    crs="EPSG:4326"
)
# Extract the 'geometry' column as a GeoDataFrame for spatial operations
points = gpd.GeoDataFrame(gdf_apartments['geometry'])                                       


# Zip Code GeoData
# ------------------------------------------------------------------------------------
# Load the USA ZIP code boundaries from a geodatabase file
zip_boundaries = gpd.read_file("data_source/USA_Zip_Code_Boundaries/v10/zip_poly.gdb")
zip_points = points.sjoin(zip_boundaries, how='left')                                       # Spatial join: match apartment points with ZIP code polygons based on location
df_apartments_with_zip = df_apartments.join(zip_points)                                     # Join the ZIP code data with the original apartment data
df_apartments_with_zip.rename(columns={'ZIP_CODE': 'zipcode'}, inplace=True)                # Rename ZIP code column for clarity


# Zip Code Data
# ------------------------------------------------------------------------------------
# Load IRS income data
df_irs = pd.read_csv("data_source/21zpallagi.csv")
df_irs = df_irs[['STATEFIPS', 'STATE', 'zipcode', 'agi_stub', 'N1']]                        # Select relevant columns from IRS dataset
zip_return_counts = df_irs.groupby('zipcode')['N1'].sum().reset_index(name='num_returns')   # Group by ZIP code to get the total number of tax returns filed in each ZIP code

df_irs = pd.merge(df_irs, zip_return_counts, on='zipcode', how='left')                      # Merge the return counts back with the IRS data
df_irs['agi_stub_perc'] = df_irs['N1'] / df_irs['num_returns']                              # Calculate the percentage of returns for each AGI (Adjusted Gross Income) stub
df_irs_agg = df_irs.groupby(['zipcode', 'agi_stub'])['agi_stub_perc'].mean().reset_index()  # Aggregate by ZIP code and AGI stub to get the mean percentage of returns per range

# Reshape the data to have AGI stub ranges as separate columns, rename the columns to reflect income ranges more clearly
df_irs_pivot = df_irs_agg.pivot(index='zipcode', columns='agi_stub', values='agi_stub_perc').reset_index()
df_irs_pivot.rename({
    1: "perc_sub25k",
    2: "perc_25_50k",
    3: "perc_50_75k",
    4: "perc_75_100k",
    5: "perc_100_200k",
    6: "perc_abv200k"
}, axis='columns', inplace=True)

# Ensure ZIP codes are treated as strings for both datasets
df_apartments_with_zip['zipcode'] = df_apartments_with_zip['zipcode'].astype(str)
df_irs_pivot['zipcode'] = df_irs_pivot['zipcode'].astype(str)

# Merge apartment data with IRS income distribution data
df_final = pd.merge(df_apartments_with_zip, df_irs_pivot, on='zipcode', how='left')

# Save the merged dataset to a CSV file
df_final.to_csv('data_modified/merged_data.csv', encoding='utf-8', index=False)