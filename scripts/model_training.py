"""This scripts does the model training.

Example usage:
    python3 model_training.py --dataset ca_stockton.csv \
        --output_class is_arrested \
        --algorithm LR
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PREPROCESSED_DATASET_DIRECTORY = '../datasets/preprocessed'


def print_roc(model, X_test, y_test) -> None:
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc_score = roc_auc_score(y_test, y_score)

    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def print_results(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


def main(dataset: str, output_class: str, algorithm: str) -> None:
    dataset_path = Path(PREPROCESSED_DATASET_DIRECTORY) / dataset

    df = pd.read_csv(dataset_path)

    output_class_with_features = {
        "is_arrested": ["driver_race", "driver_gender", "search_conducted", "age_group"],
        "driver_gender": ["driver_race", "is_arrested", "search_conducted", "stop_outcome", "age_group"],
        "age_group": ["driver_race", "driver_gender", "search_conducted", "stop_outcome", "is_arrested"],
    }

    X = df[output_class_with_features[output_class]]
    X = pd.get_dummies(X, columns=X.columns)
    y = df[output_class]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if output_class == 'is_arrested' or output_class == 'age_group':
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        "LR": LogisticRegression(solver='newton-cg'),
        "DTC": DecisionTreeClassifier(),
        "RFC": RandomForestClassifier(),
        "NB": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    model = models.get(algorithm)
    model.fit(X_train, y_train)

    # Train Predict
    y_train_pred = model.predict(X_train)
    print('\nTrain Results')
    print_results(y_train, y_train_pred)

    # Test Predict
    y_test_pred = model.predict(X_test)
    print('\nTest Results')
    print_results(y_test, y_test_pred)

    if output_class == 'driver_gender' or output_class == 'is_arrested':
        print_roc(model, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_class', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)

    args = parser.parse_args()
    main(args.dataset, args.output_class, args.algorithm)
