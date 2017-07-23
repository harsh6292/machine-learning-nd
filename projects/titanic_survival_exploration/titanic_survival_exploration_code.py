#!/usr/bin/python

import numpy as np
import pandas as pd
import visuals as vs

from IPython.display import display

#%matplotlib_inline

# Read the dataset containing titanic info
file = 'titanic_data.csv'
titanic_data = pd.read_csv(file)

# Print the first few lines from data
#display(titanic_data.head())

# Store and remove survival rate from data
survived_data = titanic_data['Survived']
titanic_data = titanic_data.drop('Survived', axis =1)

#display(titanic_data.head())
#display(survived_data.head())

def accuracy_score(truth, predicted):
    
    if len(truth) == len(predicted):
        return "Predictions have an accuracy of {:.2f}%.".format((truth == predicted).mean()*100)
    else:
        return "Number of predictions don't match with number of outcomes."

predictions = pd.Series(np.ones(5, dtype = int))
print "Accuracy score\n"
print accuracy_score(survived_data[:5], predictions)

# Our own prediction model, it always predicts that no passenger survived
def predictions_0(data):
    predictions = []
    
    for _, passenger in data.iterrows():
        # Predict passenger didn't survived
        predictions.append(0)
    
    return pd.Series(predictions)
    
predictions = predictions_0(titanic_data)

# Print the accuracy score based on our prediction model
# All predictions are zero, but prediction accuracy will be one where 0 matches survival data
print accuracy_score(survived_data, predictions)

# Now lets see the effect of gender on predictions
# We will predict that all females have survived
def predictions_1(data):
    predictions = []
    
    for _, passenger in data.iterrows():
        if passenger['Sex'] == "female":
            predictions.append(1)
        else:
            predictions.append(0)

    return pd.Series(predictions)

predictions = predictions_1(titanic_data)

# Print the accuracy of survival rate if only females survived
print accuracy_score(survived_data[:6], predictions[:6])


# Now lets consider males younger than 10 have survived. and add it into our prediction model
def predictions_2(data):
    predictions = []
    
    for _, passenger in data.iterrows():
        if passenger['Sex'] == "female":
            predictions.append(1)
        else:
            if passenger['Age'] <= 10:
                predictions.append(1)
            else:
                predictions.append(0)

    return pd.Series(predictions)

predictions = predictions_2(titanic_data)
# Print the accuracy of survival rate
print accuracy_score(survived_data, predictions)

# Refine our logic to make predictions greater than 80%
def predictions_3(data):
    predictions = []
    
    for _, passenger in data.iterrows():
        if passenger['Sex'] == "female":
            if passenger['Pclass'] == 1:
                predictions.append(1)
            elif passenger['SibSp'] <= 2:
                predictions.append(1)
            elif passenger['Parch'] == 0:
                predictions.append(1)
            elif passenger['Fare'] >= 100:
                predictions.append(1)
            else:
                predictions.append(0)
        else:
            if passenger['Age'] <= 10:
                predictions.append(1)
            elif passenger['Pclass'] == 1 and passenger['Parch'] >= 2 and passenger['SibSp'] == 1:
                predictions.append(1)
            else:
                predictions.append(0)

    return pd.Series(predictions)

predictions = predictions_3(titanic_data)
# Print the accuracy of survival rate
print "Accuracy score based on various parameters \n"
print accuracy_score(survived_data, predictions)
    

