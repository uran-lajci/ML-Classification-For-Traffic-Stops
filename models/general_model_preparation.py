import pandas as pd
import numpy as np


def show_dataset_information(df):
    print("Dataset information before preprocessing\n")
    print("General information")
    print(df.info())
    print("\n")
    print("The number of rows: ", df.shape[0])
    print("Number of columns: ", df.shape[1])
    print("The number of null values in columns")
    print(df.isna().sum())


def read_dataset(choose_dataset):
    if choose_dataset == "a":
        df = pd.read_csv('./datasets/training/initial_police_dataset.csv')
        df = df[['driver_gender', 'driver_age_raw', 'driver_race', 'stop_outcome', 'is_arrested', "stop_date", "stop_time", "driver_age"]]
        # The rows with missing values in all these columns should be deleted because they do not contain enough information to make a prediction
        df = df.dropna(subset=['driver_gender', 'driver_age_raw', 'driver_race', 'stop_outcome', 'is_arrested'])
        return df
    
    elif choose_dataset == "b":
        df = pd.read_csv('./datasets/retraining/nc_durham_2020_04_01.csv')
        df = df[["subject_sex", "subject_race", "outcome", "arrest_made", "date", "time", "subject_age"]]
        df.rename(columns={'subject_sex': 'driver_gender', 'subject_race': 'driver_race', 'stop_outcome': 'outcome',
                            'is_arrested': 'arrest_made', 'is_arrested': 'arrest_made'}, inplace=True)
        return df

    elif choose_dataset == "c":
        df = pd.read_csv('./datasets/retraining/ri_statewide_2020_04_01.csv')
        df = df[["subject_sex", "subject_race", "outcome", "arrest_made", "date", "time", "subject_age"]]
        return df

    elif choose_dataset == "d":
        df = pd.read_csv('./datasets/retraining/vt_burlington_2023_01_26.csv')
        df = df[["subject_sex", "subject_race", "outcome", "arrest_made", "date", "time", "subject_age"]]
        return df
    else:
        print("/nChoose one model to preprocess, the options are a, b, c, d")
        exit()


def get_the_date_time(df):
    # We combine the date and time in one column and make their type datetime
    df['stop_datetime'] = pd.to_datetime(df['stop_date'] + ' ' + df['stop_time'])

    # Since we have the stop date and stop time in the new column stop_datetime we do not need stop_date and stop_time so we delete them
    df = df.drop(['stop_date', 'stop_time'], axis=1)
    return df


def get_the_cleaned_ages(df):
    # Replacing anomaly years in driver age raw
    valid_years = df[(df['driver_age_raw'] >= 1900) & (df['driver_age_raw'] <= df['stop_datetime'].dt.year.max())]
    mean_year = int(np.mean(valid_years['driver_age_raw']))
    df.loc[(df['driver_age_raw'] < 1900) | (df['driver_age_raw'] > df['stop_datetime'].dt.year.max()), 'driver_age_raw'] = mean_year

    # We make this fill only if the difference between stop_datetime year and driver_age raw is bigger than 16
    df['driver_age'] = np.where((df['stop_datetime'].dt.year - df['driver_age_raw']) > 16, df['driver_age'].fillna(df['stop_datetime'].dt.year - df['driver_age_raw']), df['driver_age'])

    # Since we made all the changes to fill the driver age so the remaining records contain wrong data, so we delete them
    df = df.dropna(subset=['driver_age'])
    return df


def get_the_discretized_ages(df):
    # Discretization of the driver's age
    # Define the categories for age groups
    bins = [14, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Define the labels for age groups
    labels = ['14-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    # Create a new column 'age_group' with the age group labels
    df['age_group'] = pd.cut(df['driver_age'], bins=bins, labels=labels)
    return df

if __name__ == "__main__":
    df = read_dataset("a")
    
    show_dataset_information(df)

    df = get_the_date_time(df)
    df = get_the_cleaned_ages(df)
    df = get_the_discretized_ages(df)

    # Remove duplicates
    df = df.drop_duplicates()

    # Print information about the preprocessed data
    print("\nData after preprocessing:")
    print(df.info())

    # Count the number of null values in each column
    print("\nThe number of null values in columns:")
    print(df.isna().sum())
    
    # Print the first five rows of the new dataset
    print(df.head())