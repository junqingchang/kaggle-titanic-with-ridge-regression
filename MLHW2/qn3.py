import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_data = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
X_valid = X_data.sample(frac=0.2,random_state=200)
X_train = X_data.drop(X_valid.index)
Y_data = X_data["Survived"]
Y_valid = X_valid["Survived"]
Y_train = X_train["Survived"]
ID_test = X_test["PassengerId"]

from IPython.display import display
display(X_data.head())
display(X_data.describe())
display(X_test.head())
display(X_test.describe())

# Commented out as process is automated using function preprocess
# df = X_train
# df.drop(["Survived"],axis=1,inplace=True,errors="ignore")
# df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)

# df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
# df["Fare"].fillna(df["Fare"].median() ,inplace=True)
# df["Age"].fillna(df["Age"].mean() ,inplace=True)

# df = df.join(pd.get_dummies(df["Embarked"]))
# df.drop(["Embarked"],axis=1,inplace=True)
# df = df.join(pd.get_dummies(df["Sex"]))
# df.drop(["Sex"],axis=1,inplace=True)
# df = df.join(pd.get_dummies(df["Pclass"]))
# df.drop(["Pclass"],axis=1,inplace=True)

# df["Family"] = df.apply(lambda row: 1
#     if row["SibSp"] != 0 or row["Parch"] != 0
#     else 0, axis=1)
# df["Child"] = df.apply(lambda row: 1
#     if row["Age"] < 16
#     else 0, axis=1)

def preprocess(df):
    df.drop(["Survived"],axis=1,inplace=True,errors="ignore")
    df.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)

    df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
    df["Fare"].fillna(df["Fare"].median() ,inplace=True)
    df["Age"].fillna(df["Age"].mean() ,inplace=True)

    df = df.join(pd.get_dummies(df["Embarked"]))
    df.drop(["Embarked"],axis=1,inplace=True)
    df = df.join(pd.get_dummies(df["Sex"]))
    df.drop(["Sex"],axis=1,inplace=True)
    df = df.join(pd.get_dummies(df["Pclass"]))
    df.drop(["Pclass"],axis=1,inplace=True)

    df["Family"] = df.apply(lambda row: 1
        if row["SibSp"] != 0 or row["Parch"] != 0
        else 0, axis=1)
    df["Child"] = df.apply(lambda row: 1
        if row["Age"] < 16
        else 0, axis=1)

    # Own tweak here removing unrelated things
    df.drop(["C","Q","S"],axis=1, inplace=True)
    return df

# Part (a)
X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_data = preprocess(X_data)
X_test = preprocess(X_test)
display(X_train.head())

# Part (b)
model = LogisticRegression(fit_intercept=False)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_valid)
print("Accuracy = {acc}".format(acc = accuracy_score(Y_valid, Y_pred)))
# Accuracy score is 0.78651

# Part (c)
partc = LogisticRegression(fit_intercept=False)
partc.fit(X_data, Y_data)
print("Coefficient = {coeff}".format(coeff=partc.coef_))
# Coefficient = [[-0.02176884 -0.59962491 -0.31966545  0.00318019  0.27527381  0.38966949
#   -0.01322367  1.62504678 -0.97332714  1.09627591  0.31154606 -0.75610233
#    0.6960641   1.20455033]]

Y_test = partc.predict(X_test)
ans = pd.DataFrame({"PassengerId":ID_test,"Survived":Y_test})
ans.to_csv("submit.csv", index=False)
# Score 0.77990
# User ID 2327696