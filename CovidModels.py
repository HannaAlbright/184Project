import pandas as pd
from sklearn import metrics, preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
import numpy as np

def CNB(xTrain, xTest, yTrain, yTest):
    model = CategoricalNB()
    model.fit(xTrain, yTrain)
    print(xTest.shape)
    yPred = model.predict(xTest)
    print("Precision:", metrics.precision_score(yTest, yPred))
    print("Recall:", metrics.recall_score(yTest, yPred))
    print("F1:", metrics.f1_score(yTest, yPred))
    print("Test Accuracy: ", metrics.accuracy_score(yTest, yPred))
    return np.concatenate((yTest, yPred), axis=0)

for filenum in [1,2,3,4,5,6,7,8,9]:
    dset="CovData/df" + str(filenum)+".csv"
    df=pd.read_csv(dset)
    print("\n", dset, " Shape: ", df.shape)
    x=df.drop(["death_yn"], axis=1)
    x=x.astype("category")
    x2 = pd.get_dummies(x)
    xOHE = x2.to_numpy()
    y = df["death_yn"].values
    le = preprocessing.LabelEncoder()
    yLE = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(xOHE, yLE, test_size=0.30, random_state=123, stratify=yLE) #using stratify since its unbalanced
    result=CNB(x_train, x_test, y_train, y_test)
