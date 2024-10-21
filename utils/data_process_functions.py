import os
import sqlite3
import pandas as pd
import numpy as np


#########################---------------------------------------------------------------------

# This script contains functions for data processing and analysis.
# Ensure this file is in the same directory as your project. 

#-----------------------------------------------------------------------------------------------

#####################-----------------------------------
## DATA LABELS
#####################-----------------------------------

# Objective: Easy data manipulation.
def data_labels(access_df, five_hundred_df):
    ## Only factors suchs as Household income and chronic diseases related with food insecurity conditions had been considered 
    # to perform the PCA and clustering, due to their highly correlation with food desserts enviorments.
    # Chronic diseases columns names
    chronic_disease_labs = [col for col in five_hundred_df.columns if 'CrudePrev' in col and 
                            any(sub in col for sub in ['DIABETES', 'BPHIGH', 'HIGHCHOL', 'STROKE', 'OBESITY', 'ACCESS', 'MHLTH'])]

    # Low Access Demographics columns names (count)
    demography_labs_cnt = [col for col in access_df.columns if '10' not in col and 'PCT' not in col and
                    any(sub in col for sub in ['CHILD', 'SENIOR', 'WHITE', 'BLACK', 'HISP', 'ASIA', 'NHNA', 'NHP'])]

    # Low Access Demographics columns names (percent)
    demography_labs_pct = [col for col in access_df.columns if '10' not in col and 'PCT' in col and
                    any(sub in col for sub in ['CHILD', 'SENIOR', 'WHITE', 'BLACK', 'HISP', 'ASIA', 'NHNA', 'NHP'])]

    # Low Access columns names (percent)
    percent_labs = [col for col in access_df.columns if 'PCT' in col and not any(sub in col for sub in demography_labs_pct + ['PCH', '10', 'MULTI'])]
    
    # Low Access columns names (count)
    count_labs = [col for col in access_df.columns if 'LACCESS' in col and not any(sub in col for sub in demography_labs_cnt + ['PCT', 'PCH', '10', 'MULTI'])]
    
    change_labs = [col for col in access_df.columns if 'PCH' in col]
    
    return chronic_disease_labs, demography_labs_cnt, demography_labs_pct, percent_labs, count_labs, change_labs


#-----------------------------------------------------------------------------------------------

#####################---------------------------------
## SUMMARIZING FIVE HUNDRED DATASET INTO COUNTY LEVEL
####################-----------------------------------

# Objective: Summarise five hundred cities data from Tract level to county level.
def five_hundred_cities_to_county_level(access_raw, five_hundred_cities_raw, chronic_disease_labs):
    # Convert FIPS, PlaceFIPS, and TractFIPS to strings
    access_raw['FIPS'] = access_raw['FIPS'].astype(str)
    five_hundred_cities_raw['TractFIPS'] = five_hundred_cities_raw['TractFIPS'].astype(str)

    # Identify which combination of FIPS and State has only 4-digits FIPS
    states_with_4_digit_FIPS = access_raw[access_raw['FIPS'].str.len() == 4]['State'].unique().tolist()

    # Define a function to extract digits based on the states that have 4 or 5 FIPS digits
    def extract_digits(row):
        if row['StateAbbr'] in states_with_4_digit_FIPS:
            return row['TractFIPS'][:4]  # Extract first 4 digits
        else:
            return row['TractFIPS'][:5]  # Extract first 5 digits

    # Create a new variable 'FIPS' in Five Hundred Cities Raw data based on the condition
    five_hundred_cities_raw['FIPS'] = five_hundred_cities_raw.apply(extract_digits, axis=1)

    # Merge the county variable from the Access dataset to Five Hundred Cities Raw
    five_hundred_cities = pd.merge(access_raw[['FIPS', 'County']], five_hundred_cities_raw, 
                                   how='right', left_on=['FIPS'], right_on=['FIPS'])
    
    five_hundred_cities = five_hundred_cities[['FIPS', 'StateAbbr', 'County', 'Population2010'] + chronic_disease_labs]
    
    # Getting only columns that are numeric. TractFIPS and StateAbrr are excluded
    float_columns = five_hundred_cities.select_dtypes(include= ['float64', 'number']).columns
    
    # Summarizing five_hundred_cities at county level
    five_hundred_cities_county_level = five_hundred_cities.groupby(['FIPS', 'StateAbbr', 'County'])[float_columns].median().reset_index()

    return five_hundred_cities_county_level

#-----------------------------------------------------------------------------------------------

#####################----------------------------------------------------
## IDENTIFY LOWER OUTLIERS
####################-----------------------------------------------------

# Objective: Identify extreme cases below the 0.01 percentile to drop them
def identify_and_gather_lower_outliers(df, columns, lower_percentile=0.01):
    outliers_dict = {}
    outlier_rows = pd.DataFrame()

    for col in columns:
        if col in df.columns:
            # Calculate the lower bound (e.g., 1st percentile)
            lower_bound = df[col].quantile(lower_percentile)

            # Identify outliers that are below the lower bound
            outliers = df[df[col] < lower_bound]

            # Append outliers to the combined outlier DataFrame
            outlier_rows = pd.concat([outlier_rows, outliers])

            # Store information about the outliers for the column
            outliers_dict[col] = {
                'lower_bound': lower_bound,
                'outliers': outliers
            }

    # Remove duplicate rows in case they were identified as outliers in multiple columns
    outlier_rows = outlier_rows.drop_duplicates()

    return outliers_dict, outlier_rows



#-----------------------------------------------------------------------------------------------

#####################----------------------------------------------------
## CALCULATE LOW INCOME POPULATION
####################-----------------------------------------------------

# Objective: Calculate the low-income and low access population for an specific group
# Assumption: The number of individuals in a specific group is calculated by 
# multiplying the percentage of people with low-income by the percentage of that population group, 
# and then multiplying this result by the estimated population numbe

def calculate_low_income_population(df_raw, columns_pct):
    
    df = df_raw.copy()
    
    for col in columns_pct:
        # Calculate the low-income population for each specified column
        df[f'LOWI_{col.replace("PCT_", "")}'] = np.ceil(((df[col]/100) * (df['PCT_LACCESS_LOWI15']/100)) * df['Population_Estimate_2018'])
    
    return df

#-----------------------------------------------------------------------------------------------

#####################----------------------------------------------------
## PREPARE POPULATION DATA TO COMPARE THE SUBGROUP IMPACT
####################-----------------------------------------------------


# Objective: Prepare population and demography data, merging both datasets and calculating the count of individuals for each demographic subgroup.
def load_and_prepare_population_data(population, demography_data):
    
    # Convert FIPS codes to string type for merging
    population['FIPS'] = population['FIPS'].astype(str)
    
    # Select relevant columns from population data
    population_2018 = population[['FIPS', 'Population_Estimate_2015', 'Population_Estimate_2018']]
    
    # Convert FIPS codes to string type for merging
    demography_data['FIPS'] = demography_data['FIPS'].astype(str)
    
    # Select only columns that contain 'FIPS' or 'PCT' from demography data
    demography_data = demography_data[[col for col in demography_data.columns if 'FIPS' in col or 'PCT' in col]]
    
    # Merge population data with demography data on FIPS code
    pop_and_demography = pd.merge(population_2018, demography_data, on='FIPS', how='inner')
    
    # Identify percentage columns for calculation
    pct_cols = [col for col in pop_and_demography.columns if 'PCT' in col]
    
    # Calculate the count of individuals for each demographic subgroup
    for col in pct_cols:
        pop_and_demography[f'CNT_{col.replace("PCT_", "")}'] = np.ceil((pop_and_demography[col] / 100) * pop_and_demography['Population_Estimate_2015'])
    
    return pop_and_demography