import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def clean_and_select_features(file_path):
    df = pd.read_csv(file_path)

    # Mapping column names to standardize them across datasets
    column_mapping = {
        'date': 'date',
        'stop_date': 'date',
        'time': 'time',
        'stop_time': 'time',
        'location': 'location',
        'county_name': 'location',
        'subject_age': 'age',
        'driver_age': 'age',
        'subject_race': 'race',
        'driver_race': 'race',
        'subject_sex': 'sex',
        'driver_gender': 'sex',
        'search_conducted': 'search_conducted',
        'outcome': 'outcome',
        'stop_outcome': 'outcome',
        'arrest_made': 'arrest',
        'is_arrested': 'arrest'
    }

    df.rename(columns=column_mapping, inplace=True)

    # Selecting only the columns that are common across the datasets
    df = df[['date', 'time', 'location', 'age', 'race', 'sex', 'search_conducted', 'outcome', 'arrest']]
    
    # The rows with missing values in all these columns should be deleted because they do not contain enough information to make a prediction
    df = df.dropna(subset=['date', 'time', 'location', 'age', 'race', 'sex', 'search_conducted', 'outcome', 'arrest'])

    # Cleaning steps
    df['date'] = pd.to_datetime(df['date'])
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['search_conducted'] = df['search_conducted'].astype(bool)
    df['arrest'] = df['arrest'].astype(bool)

    bins = [14, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Define the labels for age groups
    labels = ['14-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    # Create a new column 'age_group' with the age group labels
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    for col in ['race', 'sex', 'outcome']:
        df[col] = df[col].astype('category')

    return df

# Usage example:
file_path1 = './datasets/retraining/nc_durham_2020_04_01.csv'
file_path2 = './datasets/retraining/nc_raleigh_2020_04_01.csv'
file_path3 = './datasets/retraining/nc_winston-salem_2020_04_01.csv'
file_path4 = './datasets/training/initial_police_dataset.csv'

cleaned_df1 = clean_and_select_features(file_path1)
print("NC Durham Cleaned Dataset")
print(cleaned_df1.head(), "\n")
# cleaned_df1.to_csv('preprocessed_nc_durham.csv', index=False)

cleaned_df2 = clean_and_select_features(file_path2)
print("NC Raleigh Cleaned Dataset")
print(cleaned_df2.head(), "\n")
# cleaned_df2.to_csv('preprocessed_nc_raleigh.csv', index=False)

cleaned_df3 = clean_and_select_features(file_path3)
print("NC Winston Salem Cleaned Dataset")
print(cleaned_df3.head(), "\n")
# cleaned_df3.to_csv('preprocessed_nc_winston.csv', index=False)