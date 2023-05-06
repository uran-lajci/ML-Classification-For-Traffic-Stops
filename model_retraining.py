from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Read the CSV file
df = pd.read_csv("preprocessed_police_project.csv")

# Get the number of rows to use for test and train
number_of_rows_to_use_for_test_and_train = int(input(f"Write the number of rows you want to use from {len(df)} that are in the dataset: "))
rows_randomly_selected = input(f"Write y if you want to randomly select rows: ")
if rows_randomly_selected == "y":
    df  = df.sample(number_of_rows_to_use_for_test_and_train)
else:
    df = df.iloc[0:number_of_rows_to_use_for_test_and_train]

# Define a dictionary that maps each attribute to its corresponding features
attribute_to_features = {
    "stop_outcome": ['violation', 'search_conducted', 'stop_duration', 'drugs_related_stop', 'stop_datetime', 'age_group'],
    "is_arrested": ['driver_gender', 'driver_race', 'violation', 'search_conducted', 'stop_outcome', 'stop_duration', 'drugs_related_stop', 'stop_datetime', 'age_group'],
    "driver_gender": ['violation','is_arrested', 'search_conducted', 'stop_duration', 'drugs_related_stop', 'age_group'],
    "age_group": ['driver_gender', 'violation', 'search_conducted', 'stop_outcome', 'stop_duration', 'drugs_related_stop'],
    "violation": ['driver_gender','driver_race', 'search_conducted', 'stop_outcome','drugs_related_stop', 'age_group']
}

# Get the attribute to be predicted
attribute_to_be_predicted = input("Choose which attribute you want to predict from these attributes: stop_outcome, driver_gender, age_group, is_arrested or violation\nAttribute to be predicted: ")

# Check if the input is valid and extract the features and target variable
if attribute_to_be_predicted in attribute_to_features:
    X = df[attribute_to_features[attribute_to_be_predicted]]
    X = pd.get_dummies(X, columns=X.columns)
    y = df[attribute_to_be_predicted]
else:
    print(f"Error. Wrong input {attribute_to_be_predicted}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Ask user to choose the model to train
training_model = input("Choose the model to train (LR, DTC, RFC, NB, KNN): ")

# Initialize the model
models = {"LR": LogisticRegression(solver='newton-cg'),
          "DTC": DecisionTreeClassifier(),
          "RFC": RandomForestClassifier(),
          "NB": GaussianNB(),
          "KNN": KNeighborsClassifier()}

model = models.get(training_model)

if model is None:
    print(f"Error. Wrong input {training_model}")

# Train the model
model.fit(X_train, y_train)

# Predict using the trained model and calculate accuracy
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print('============================================')
print('============================================')

print('See if we have underfitting or overfitting ')

# Predict using the trained model and calculate accuracy
y_train_pred = model.predict(X_train)
train_accuracy = round(accuracy_score(y_train, y_train_pred) * 100, 2)

y_test_pred = model.predict(X_test)
test_accuracy = round(accuracy_score(y_test, y_test_pred) * 100, 2)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Use cross-validation to evaluate the model
cv_scores = cross_val_score(model, X, y, cv=5)
cv_accuracy = round(cv_scores.mean() * 100, 2)
cv_std = round(cv_scores.std() * 100, 2)

print(f"Cross-validation Accuracy: {cv_accuracy} (+/- {cv_std})")
print('============================================')

# Print the accuracy, confusion matrix and classification report of the model
print(f"MODEL : {accuracy}")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print('================')
print(classification_report(y_test, y_pred))
print('====================================================')
print('====================================================')

# Use cross-validation to evaluate the model
scores = cross_val_score(model, X, y, cv=5)

# Print the average accuracy and standard deviation across folds
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std()*2:.2f})")

if (attribute_to_be_predicted == 'driver_gender' or attribute_to_be_predicted == 'is_arrested'):
    # Convert categorical labels to binary labels
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)

    # Get the predicted probabilities
    y_score = model.predict_proba(X_test)[:, 1]

    # Compute the ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    auc_score = roc_auc_score(y_test, y_score)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()