import pandas as pd
import numpy as np

# Predicting whether a person will be arrested and te outcome of the stop based on their data
# Phase 1 - Model preparation

df = pd.read_csv('./datasets/training/initial_police_dataset.csv')

print("Dataset information before preprocessing\n")
print("General information")
print(df.info())
print("\n")
print("The number of rows: ", df.shape[0])
print("Number of columns: ", df.shape[1])
print("The number of null values in columns")
print(df.isna().sum())

# We delete the county_name column from the dataset since all the values of this column are missing
df = df.drop(['county_name','search_type'], axis=1)

# The search_type column has 88545 missing values from 91741 rows in total, but its data can be useful in our work during the analysis and prediction,
# so we replace its missing values with Unknown and do not change the not missing values.
#df['search_type'] = df['search_type'].fillna('Unknown')

# The rows with missing values in all these columns should be deleted because they do not contain enough information to make a prediction
df = df.dropna(subset=['driver_gender', 'driver_age_raw', 'driver_race', 'violation_raw', 'violation', 'stop_outcome', 'is_arrested', 'stop_duration'])

#Droping records that ar not appropriate
df = df.drop( df.index[df['stop_duration'].isin(['1', '2'])])

# We combine the date and time in one column and make their type datetime
df['stop_datetime'] = pd.to_datetime(df['stop_date'] + ' ' + df['stop_time'])

# Since we have the stop date and stop time in the new column stop_datetime we do not need stop_date and stop_time so we delete them
df = df.drop(['stop_date', 'stop_time'], axis=1)

# Replacing anomaly years in driver age raw
valid_years = df[(df['driver_age_raw'] >= 1900) & (df['driver_age_raw'] <= df['stop_datetime'].dt.year.max())]
mean_year = int(np.mean(valid_years['driver_age_raw']))
df.loc[(df['driver_age_raw'] < 1900) | (df['driver_age_raw'] > df['stop_datetime'].dt.year.max()), 'driver_age_raw'] = mean_year

# We make this fill only if the difference between stop_datetime year and driver_age raw is bigger than 16
df['driver_age'] = np.where((df['stop_datetime'].dt.year - df['driver_age_raw']) > 16, df['driver_age'].fillna(df['stop_datetime'].dt.year - df['driver_age_raw']), df['driver_age'])

# Since we made all the changes to fill the driver age so the remaining records contain wrong data, so we delete them
df = df.dropna(subset=['driver_age'])

# Discretization of the driver's age
# Define the categories for age groups
bins = [14, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Define the labels for age groups
labels = ['14-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
# Create a new column 'age_group' with the age group labels
df['age_group'] = pd.cut(df['driver_age'], bins=bins, labels=labels)

# Count the number of duplicates
print("Number of duplicates:")
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# Print the data types of each column
print("\nData types:")
print(df.dtypes)

# Print information about the preprocessed data
print("\nData after preprocessing:")
print(df.info())

# Count the number of null values in each column
print("\nThe number of null values in columns:")
print(df.isna().sum())

# Feature Selection
# Select features to be used in prediction based on intuition
columns_to_be_used_in_prediction = [
    'driver_gender', 'driver_race', 'violation', 'search_conducted',
    'stop_outcome', 'is_arrested', 'stop_duration',
    'drugs_related_stop', 'stop_datetime', 'age_group'
]

new_dataset = df[columns_to_be_used_in_prediction]

# Print the first five rows of the new dataset
print(new_dataset.head())

# Save the preprocessed dataset to a new CSV file
# new_dataset.to_csv('preprocessed_police_project.csv', index=False)