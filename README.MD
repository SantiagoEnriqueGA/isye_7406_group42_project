# ISYE 7406 Group 42 Project 

## Files in src/data_eng

### dataJoining.py

This script merges apartment rental data with geographical ZIP code boundaries and IRS income data. Key steps include:

1. **Loading Data**: Reads apartment listings and ZIP code boundaries, converting the listings to spatial data using latitude and longitude.
2. **Spatial Join**: Matches apartment listings to ZIP codes using geographical boundaries.
3. **IRS Income Data**: Aggregates IRS income distribution by ZIP code, calculating the percentage of tax returns in various income ranges.
4. **Merging**: Combines apartment data with income distribution by ZIP code.
5. **Saving**: Outputs the final merged dataset to a CSV file.

Output: `merged_data.csv`


### dataFeatureEng.py

This script performs feature engineering on apartment rental data, creating new variables, handling missing values, and preparing the dataset for modeling. Key steps include:

1. **Loading Data**: Reads the merged dataset for further processing.
2. **Feature Creation**: 
   - Calculates the length of property descriptions.
   - Creates binary variables indicating if cats or dogs are allowed.
   - Generates dummy variables for each listed amenity.
3. **City Grouping**: Groups cities with fewer than 50 listings into an "Other" category.
4. **Price Type Filtering**: Removes uncommon price types.
5. **Address Indicator**: Adds a binary column indicating the presence of an address.
6. **Column Removal**: Drops redundant columns that are not needed for analysis.
7. **Missing Value Handling**: Removes rows with missing values in key columns.
8. **Saving**: Exports the cleaned and engineered dataset to a CSV file.

Output: `engineered_data.csv`

