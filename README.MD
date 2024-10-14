# ISYE 7406 Group 42 Project 


<!-- 
Data sources:

UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified
USA ZIP Code Areas: https://www.arcgis.com/home/item.html?id=8d2012a2016e484dafaac0451f9aea24
SOI Tax Stats - Individual income tax statistics - 2021 ZIP Code data (SOI): https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2021-zip-code-data-soi-0
 -->


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

## Files in src/eda

### eda.ipynb

This notebook provides an exploratory data analysis (EDA) for the apartment rental dataset, focusing on visualizing key numerical and categorical features, identifying correlations, and examining the relationships between `price` and other important variables.

Key steps performed in the notebook include:

1. **Setting Working Directory**: The notebook adjusts the working directory to access data files from the appropriate location.

2. **Data Loading**: 
   - Reads the `engineered_data.csv` file containing the processed dataset.
   - Applies log transformations to the `price` and `square_feet` columns (`log1p` transformation: `log(1 + x)`) to reduce skewness.

3. **Basic Information and Summary Statistics**:
   - Displays basic information about the dataset such as data types, missing values, and basic statistics.

4. **Distribution of Key Numerical Features**:
   - Plots histograms with kernel density estimates for key numerical variables including `price`, `log1p_price`, `square_feet`, `log1p_square_feet`, population, and population density.

5. **Distribution of Income Percentage Features**:
   - Visualizes the distribution of income brackets across different percentage features (`perc_sub25k`, `perc_25_50k`, etc.).

6. **Categorical Feature Analysis**:
   - Uses count plots to analyze the distribution of key categorical variables such as `bedrooms`, `bathrooms`, and `state`.

7. **Correlation Heatmap**:
   - Calculates and visualizes the correlation matrix of numeric features using a heatmap with a mask for the upper triangle.

8. **Price vs Key Features**:
   - Generates regression plots showing the relationships between log-transformed price (`log1p_price`) and key variables such as `bedrooms`, `bathrooms`, `square_feet`, population, population density, and amenity count (`has_count`).
   - Additionally, regression plots are created to visualize the relationships between income percentage features and log-transformed price.

Output:
- **Plots**: A wide variety of histograms, count plots, correlation heatmaps, and regression plots for both numerical and categorical features, allowing for in-depth insights into the data structure and potential feature importance.


