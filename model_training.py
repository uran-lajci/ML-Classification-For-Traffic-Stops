import pandas as pd
import numpy as np
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

df = pd.read_csv("preprocessed_police_project.csv")

number_of_rows_to_use_for_test_and_train = int(input(f"Write the number of rows you want to use from {len(df)} that are in the dataset: "))
df = df[0:number_of_rows_to_use_for_test_and_train]

attribute_to_be_predicted = str(input("Chose wich attribute you want to predict from these attributes: stop_outcome, is_arrested\nAttribute to be predicted: "))

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
else:
    print(f"Error. Wrong input {attribute_to_be_predicted}")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# training_model = str(input("""Write LR for Logistic Regression,
#       DTC for Decision Tree Classifier,
#       RFC for Random Forest Classifier,
#       NB for Gaussian Naive Bayes,
#       KNN for KNeighbors Classifier
#       Model: """))

# Initialize the model
# if training_model == "LR":
#     model = LogisticRegression()
# elif training_model == "DTC":
#     model = DecisionTreeClassifier()
# elif training_model == "RFC":
#     model = RandomForestClassifier()
# elif training_model == "NB":
#     model = GaussianNB()
# elif training_model == "KNN":
#     model = KNeighborsClassifier()
# else:
#     print(f"Error. Wrong input {training_model}")



LRmodel = LogisticRegression()
DTmodel = DecisionTreeClassifier()
RFmodel = RandomForestClassifier()
NBmodel = GaussianNB()
KNNmodel = KNeighborsClassifier()
# Train the model on the training data
#model.fit(X_train, y_train)
LRmodel.fit(X_train, y_train)
DTmodel.fit(X_train, y_train)
RFmodel.fit(X_train, y_train)
NBmodel.fit(X_train, y_train)
KNNmodel.fit(X_train, y_train)

#Make predictions on the testing data
y_pred = LRmodel.predict(X_test)
accuracyLR = round(accuracy_score(y_test, y_pred)*100,2)
print("Logistic Regression : %f" %accuracyLR)
print(confusion_matrix(y_test, y_pred))
print('================')
print(classification_report(y_test, y_pred))
print('====================================================')
print('====================================================')


y_pred = DTmodel.predict(X_test)
accuracyDT = round(accuracy_score(y_test, y_pred)*100,2)
print("Decision Tree : %f" %accuracyDT)
print(confusion_matrix(y_test, y_pred))
print('================')
print(classification_report(y_test, y_pred))
print('====================================================')
print('====================================================')

y_pred = RFmodel.predict(X_test)
accuracyRF = round(accuracy_score(y_test, y_pred)*100,2)
print("Random Forest : %f" %accuracyRF)
print(confusion_matrix(y_test, y_pred))
print('================')
print(classification_report(y_test, y_pred))
print('====================================================')
print('====================================================')


y_pred = NBmodel.predict(X_test)
accuracyNB = round(accuracy_score(y_test, y_pred)*100,2)
print("Naive Bayes : %f" %accuracyNB)
print(confusion_matrix(y_test, y_pred))
print('================')
print(classification_report(y_test, y_pred))
print('====================================================')
print('====================================================')

y_pred = KNNmodel.predict(X_test)
accuracyKNN = round(accuracy_score(y_test, y_pred)*100,2)
print("KNN : %f" %accuracyKNN)
print(confusion_matrix(y_test, y_pred))
print('================')
print(classification_report(y_test, y_pred))
print('====================================================')
print('====================================================')

# y_pred = model.predict(X_test)
# accuracy = round(accuracy_score(y_test, y_pred)*100,2)
# print("KNN : %f" %accuracy)
# print(confusion_matrix(y_test, y_pred))
# print('================')
# print(classification_report(y_test, y_pred))
# print('====================================================')
# print('====================================================')

# Evaluate the model's performance