"""This scripts does the preprocessing of the datasets.

Example usage:
    python3 preprocessor.py --dataset ca_stockton.csv
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

INITIAL_DATASET_DIRECTORY = '../datasets/initial'
PREPROCESSED_DATASET_DIRECTORY = '../datasets/preprocessed'


def main(dataset_name: str) -> None:
    dataset_path = Path(INITIAL_DATASET_DIRECTORY) / dataset_name

    df = pd.read_csv(dataset_path)
    df = df[['subject_sex', 'subject_race', 'arrest_made', 'search_conducted', 'outcome', 'subject_age']]

    column_mapping = {
        'subject_sex': 'driver_gender',
        'subject_race': 'driver_race',
        'arrest_made': 'is_arrested',
        'outcome': 'stop_outcome',
        'subject_age': 'age'
    }

    df.rename(columns=column_mapping, inplace=True)

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['search_conducted'] = df['search_conducted'].astype(bool)
    df['is_arrested'] = df['is_arrested'].astype(bool)

    bins = [10, 30, 50, 100]
    labels = ['10-30', '31-50', '51-100']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    df.drop('age', axis=1, inplace=True)

    for col in ['driver_race', 'driver_gender', 'stop_outcome']:
        df[col] = df[col].astype('category')

    df.dropna(inplace=True)

    output_path = Path(PREPROCESSED_DATASET_DIRECTORY) / dataset_name
    df.to_csv(output_path)

    print(f'Saved the preprocessed dataset {dataset_name}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    main(args.dataset_name)
