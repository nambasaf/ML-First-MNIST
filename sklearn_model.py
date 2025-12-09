from sklearn.datasets import load_digits
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# first thing is to load the data
digits = load_digits()

# next thing is to extract the features and labels of the data
X = digits.data
y = digits.target

print("Data shape: ", X.shape)

# Now we split the data in a ratio of train and split 
# how we do that is by using sklearn's function of train_test_split
X_train, X_test,  y_train, y_test = train_test_split(X, y ,test_size=0.4, random_state=42)

# create the model you want to use for this 
clf = LogisticRegression(max_iter=2000)

# Train on train data
clf.fit(X_train, y_train)

# get the predictions -> test on test_data
preds = clf.predict(X_test)

# check accuracy of our model 
acc = accuracy_score(y_test, preds)

print("Scikit-learn model accuracy:", acc)

