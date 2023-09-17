"""This scripts exploratory data analysis for the datasets.

Example usage:
    python3 EDA.py --dataset ca_stockton.csv
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

PREPROCESSED_DATASET_DIRECTORY = '../datasets/preprocessed'


def main(dataset_name: str) -> None:
    dataset_path = Path(PREPROCESSED_DATASET_DIRECTORY) / dataset_name

    df = pd.read_csv(dataset_path)

    print(f'Dataset {dataset_name} information.')
    print(df.info())

    print('Value counts')
    print(df['driver_gender'].value_counts(), '\n')
    print(df['driver_race'].value_counts(), '\n')
    print(df['is_arrested'].value_counts(), '\n')
    print(df['search_conducted'].value_counts(), '\n')
    print(df['stop_outcome'].value_counts(), '\n')
    print(df['age_group'].value_counts(), '\n')

    sns.countplot(x='driver_gender', data=df)
    plt.title('Count of Stops by Gender')
    plt.show()

    sns.countplot(x='driver_race', data=df)
    plt.title('Count of Stops by Race')
    plt.xticks(rotation=45)
    plt.show()

    sns.countplot(x='age_group', data=df)
    plt.title('Count of Stops by Age Group')
    plt.show()

    sns.countplot(x='stop_outcome', hue='driver_gender', data=df)
    plt.title('Stop Outcome by Gender')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()
    main(args.dataset_name)
