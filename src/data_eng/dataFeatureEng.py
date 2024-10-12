import pandas as pd

# Load the merged data from CSV file
df = pd.read_csv(
    'data_modified/merged_data.csv', 
    encoding='utf-8',
    index_col=0, # First column has the index
    low_memory=False
)

# Create a new column representing the length of each property description
# ------------------------------------------------------------------------------------
df['description_len'] = df['body'].apply(len)


# Create dummy variables to indicate whether cats or dogs are allowed
# ------------------------------------------------------------------------------------
df['cats_allowed'] = df['pets_allowed'].apply(lambda x: 1 if isinstance(x, str) and 'Cats' in x else 0)
df['dogs_allowed'] = df['pets_allowed'].apply(lambda x: 1 if isinstance(x, str) and 'Dogs' in x else 0)


# Create dummy variables for each listed amenity
# ------------------------------------------------------------------------------------
amenities_series = pd.Series(df['amenities']).astype(str)                                   # Convert amenities column to string for processing
amenities_split = amenities_series.str.split(",").explode()                                 # Split amenities by commas and "explode" into separate rows
unique_amenities = pd.Series(amenities_split).unique()                                      # Get unique amenities
amenity_colnames = pd.Series(unique_amenities).str.replace(" ", "_").str.replace("/", "_")  # Replace spaces and slashes for column names
amenity_colnames = "has_" + amenity_colnames                                                # Prefix with 'has_' to form final column names

original_columns = df.columns.tolist()                                                      # Save the existing column names

# For each unique amenity, create a new column where 1 indicates presence of the amenity, 0 otherwise
for amenity in unique_amenities:
    df[amenity] = pd.Series(df['amenities']).fillna('').str.contains(amenity).astype(int)

df.columns = original_columns + amenity_colnames.tolist()                                   # Rename the newly added columns to reflect the amenity names


# Group cities with fewer than 50 listings into an 'Other' category
# ------------------------------------------------------------------------------------
city_counts = df['cityname'].value_counts()                     # Count occurrences of each city
small_cities = city_counts[city_counts < 50].index              # Identify cities with fewer than 50 listings
df.loc[df['cityname'].isin(small_cities), 'cityname'] = 'Other' # Assign these small cities to 'Other'


# Filter price types and remove rows with uncommon price types
# ------------------------------------------------------------------------------------
df['price_type'].unique()                                       # Display the unique values in the 'price_type' column

# Count the number of rows where 'price_type' is 'Monthly|Weekly' or 'Weekly'
print(df[df['price_type'] == 'Monthly|Weekly'].shape[0])
print(df[df['price_type'] == 'Weekly'].shape[0])

# Since there are only 4 rows with these price types, drop them
df = df[~df['price_type'].isin(['Monthly|Weekly', 'Weekly'])]


# Create a new column 'has_address', set to 1 if there's an address, else 0
# ------------------------------------------------------------------------------------
df['has_address'] = df['address'].apply(lambda x: 0 if pd.isna(x) else 1)

# Drop redundant columns that are no longer needed for analysis
# ------------------------------------------------------------------------------------
columns_to_drop = [
    'category', 'amenities', 'price_type', 'pets_allowed', 'time', 'currency', 
    'price_display', 'index_right', 'PO_NAME', 'Shape_Length', 'Shape_Area', 
    'has_nan', 'STATE', 'fee', 'SQMI', 'geometry', 'address', 'body', 'title',
    'latitude','longitude'
]
df = df.drop(columns=columns_to_drop)


# Drop rows with missing values in key columns
# ------------------------------------------------------------------------------------
# Key columns to check for missing values
columns_to_check = [
    'price', 'zipcode', 'perc_sub25k', 'bedrooms', 
    'bathrooms', 'POPULATION', 'state'
]
# Drop rows with missing values in any of the specified columns
df = df.dropna(subset=columns_to_check) 


print(df.isnull().sum())

# Save the cleaned and processed data to a new CSV file
# ------------------------------------------------------------------------------------
df.to_csv(
    'data_modified/engineered_data.csv', 
    encoding='utf-8',
    index=False  # Do not write row indices to the CSV file
)
