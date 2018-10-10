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

# Follow Instructions Code
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
# =============================================

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
    return df

X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_data = preprocess(X_data)
X_test = preprocess(X_test)
display(X_train.head())

model = LogisticRegression(fit_intercept=False)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_valid)
print("Accuracy = {acc}".format(acc = accuracy_score(Y_valid, Y_pred)))

partc = LogisticRegression(fit_intercept=False)
partc.fit(X_data, Y_data)

print("Coefficient = {coeff}".format(coeff=partc.coef_))