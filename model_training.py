import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("preprocessed_police_project.csv")

number_of_rows_to_use_for_test_and_train = int(input(f"Write the number of rows you want to use from {len(df)} that are in the dataset: "))
df = df[0:number_of_rows_to_use_for_test_and_train]

attribute_to_be_predicted = str(input("Chose wich attribute you want to predict from these attributes: stop_outcome, driver_gender, age_group, is_arrested\nAttribute to be predicted: "))

if attribute_to_be_predicted == "stop_outcome":
    # Extract the features and target variable
    X = df[['driver_gender', 'driver_race', 'violation', 'search_conducted',
                'is_arrested', 'stop_duration', 'drugs_related_stop', 'stop_datetime', 'age_group']]
    y = df['stop_outcome']

    # Encode categorical variables as numbers
    X = pd.get_dummies(X, columns=['driver_gender', 'driver_race', 'violation', 'is_arrested', 'stop_duration', 'stop_datetime', 'age_group'])
elif attribute_to_be_predicted == "is_arrested":
    # Extract the features and target variable
    X = df[['driver_gender', 'driver_race', 'violation', 'search_conducted',
                'stop_outcome', 'stop_duration', 'drugs_related_stop', 'stop_datetime', 'age_group']]
    y = df['is_arrested']

    # Encode categorical variables as numbers
    X = pd.get_dummies(X, columns=['driver_gender', 'driver_race', 'violation', 'stop_outcome', 'stop_duration', 'stop_datetime', 'age_group'])
elif attribute_to_be_predicted == "driver_gender":
    # Extract the features and target variable
    X = df[[ 'driver_race', 'violation','is_arrested', 'search_conducted',
                'stop_outcome', 'stop_duration', 'drugs_related_stop', 'stop_datetime', 'age_group']]
    y = df['driver_gender']

    # Encode categorical variables as numbers
    X = pd.get_dummies(X, columns=[ 'driver_race', 'violation','is_arrested', 'stop_outcome', 'stop_duration', 'stop_datetime', 'age_group'])
elif attribute_to_be_predicted == "age_group":
    # Extract the features and target variable
    X = df[[ 'driver_gender','driver_race', 'violation','is_arrested', 'search_conducted',
                'stop_outcome', 'stop_duration', 'drugs_related_stop', 'stop_datetime', ]]
    y = df['age_group']

    # Encode categorical variables as numbers
    X = pd.get_dummies(X, columns=[ 'driver_gender','driver_race', 'violation','is_arrested', 'stop_outcome', 'stop_duration', 'stop_datetime'])
else:
    print(f"Error. Wrong input {attribute_to_be_predicted}")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

training_model = str(input("""Write LR for Logistic Regression,
      DTC for Decision Tree Classifier,
      RFC for Random Forest Classifier,
      NB for Gaussian Naive Bayes,
      KNN for KNeighbors Classifier
      Model: """))

#Initialize the model
if training_model == "LR":
    model = LogisticRegression(solver='newton-cg')
elif training_model == "DTC":
    model = DecisionTreeClassifier()
elif training_model == "RFC":
    model = RandomForestClassifier()
elif training_model == "NB":
    model = GaussianNB()
elif training_model == "KNN":
    model = KNeighborsClassifier()
else:
    print(f"Error. Wrong input {training_model}")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred)*100,2)
print("MODEL : %f" %accuracy)
print(confusion_matrix(y_test, y_pred))
print('================')
print(classification_report(y_test, y_pred))
print('====================================================')
print('====================================================')
scores = cross_val_score(model, X, y, cv=5)

# Print the average accuracy and standard deviation across folds
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))






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
