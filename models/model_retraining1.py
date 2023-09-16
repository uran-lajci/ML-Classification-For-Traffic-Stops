from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Read the CSV file
#df = pd.read_csv("./datasets/training/preprocessed_police.csv")
#df = pd.read_csv("./datasets/retraining/preprocessed/preprocessed_nc_winston.csv")
#df = pd.read_csv("./datasets/retraining/preprocessed/preprocessed_nc_raleigh.csv")
df = pd.read_csv("C:\\Users\\Elvir Misini\\Desktop\\masterFK\\S2\\ML\\Policing-ML-Model\\datasets\\training\\preprocessed_ca_stockton-3.csv")
df.dropna(subset=['age_group'], inplace=True)

#print(df.info())
# Get the number of rows to use for test and train
number_of_rows_to_use_for_test_and_train = int(input(f"Write the number of rows you want to use from {len(df)} that are in the dataset: "))
rows_randomly_selected = input(f"Write y if you want to randomly select rows: ")
if rows_randomly_selected == "y":
    df  = df.sample(number_of_rows_to_use_for_test_and_train)
else:
    df = df.iloc[0:number_of_rows_to_use_for_test_and_train]

# Define a dictionary that maps each attribute to its corresponding features
attribute_to_features = {
    "is_arrested": ["driver_race","driver_gender","search_conducted","age_group"],
    "driver_gender": ["driver_race","is_arrested","search_conducted","stop_outcome","age_group"],
    "age_group": ["driver_race","driver_gender","search_conducted","stop_outcome","is_arrested"],
}

# Get the attribute to be predicted
attribute_to_be_predicted = input("Choose which attribute you want to predict from these attributes: driver_gender, age_group or is_arrested\nAttribute to be predicted: ")

# Check if the input is valid and extract the features and target variable
if attribute_to_be_predicted in attribute_to_features:
    X = df[attribute_to_features[attribute_to_be_predicted]]
    X = pd.get_dummies(X, columns=X.columns)
    y = df[attribute_to_be_predicted]
else:
    print(f"Error. Wrong input {attribute_to_be_predicted}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# X and y are your feature matrix and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


print(y_train.value_counts())
print(y_train_resampled.value_counts())
# Ask user to choose the model to train
training_model = input("Choose the model to train (LR, DTC, RFC, NB, KNN): ")

# Initialize the model
models = {"LR": LogisticRegression(solver='newton-cg'),
          "DTC": DecisionTreeClassifier(),
          "RFC": RandomForestClassifier(),
#          "NB": GaussianNB(),
          "NB": GaussianNB(),
          "KNN": KNeighborsClassifier()}

model = models.get(training_model)

if model is None:
    print(f"Error. Wrong input {training_model}")

# Train the model
model.fit(X_train_resampled, y_train_resampled)

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

# Calculate and print the metrics
accuracy = accuracy_score(y_test, y_pred)
#cv_scores = cross_val_score(model, y_test, y_pred, cv=5)
#cv_accuracy = round(cv_scores.mean() * 100, 2)
precision = precision_score(y_test, y_pred, average='weighted') # Adjust 'average' parameter if it is a multi-class classification
recall = recall_score(y_test, y_pred, average='weighted') # Adjust 'average' parameter if it is a multi-class classification
f1 = f1_score(y_test, y_pred, average='weighted') # Adjust 'average' parameter if it is a multi-class classification

#print(f"Cross Validation 5 folds: {cv_accuracy:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


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