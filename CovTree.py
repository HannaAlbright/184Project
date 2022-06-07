import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def covTree(x_train, x_test, y_train, y_test,tNum):
    xDepth=[]
    y1Acc=[]
    y2Acc=[]
    for mDepth in range(2, 21, 2):
        pidTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=mDepth, random_state=1) #ID3 uses entropy
        pidTree.fit(x_train, y_train)
        trainPrediction = pidTree.predict(x_train)
        trainAcc=metrics.accuracy_score(y_train, trainPrediction)
        testPrediction = pidTree.predict(x_test)
        testAcc=metrics.accuracy_score(y_test, testPrediction)
        xDepth.append(mDepth)
        y1Acc.append(trainAcc)
        y2Acc.append(testAcc)
    plt.plot(xDepth, y1Acc,'g')
    plt.plot(xDepth, y2Acc, 'r')
    fname ="treeAccuracy/tree"+tNum+".png"
    plt.savefig(fname)
    plt.close()
    return pidTree

def treeDepth(x_train, x_test, y_train, y_test,mDepth):
    pidTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=mDepth, random_state=1) #ID3 uses entropy
    pidTree.fit(x_train, y_train)  # your model
    trainPrediction = pidTree.predict(x_train)
    trainAcc=metrics.accuracy_score(y_train, trainPrediction)
    testPrediction = pidTree.predict(x_test)
    testAcc=metrics.accuracy_score(y_test, testPrediction)
    print("Test Accuracy: ", testAcc)
    print("F1:", metrics.f1_score(y_test, testPrediction))
    print("Precision:", metrics.precision_score(y_test, testPrediction))
    print("Recall:", metrics.recall_score(y_test, testPrediction))
    tn, fp, fn, tp = confusion_matrix(y_test,testPrediction).ravel()
    print("TN: ", tn, " FP: ", fp, " FN: ", fn, " TP: ", tp)
    return pidTree

Xdfs=[]
ys=[]
num=0
covidFiles=["CovData/df10.csv","CovData/dftree1.csv", "CovData/dftree2.csv","CovData/dftree3.csv","CovData/dftree4.csv","CovData/dftree5.csv"]
for dset in covidFiles:
    df=pd.read_csv(dset)
    x=df.drop("death_yn", axis=1)
    x = x.astype("category")
    x2 = pd.get_dummies(x)  # county code wont do OHE!
    xOHE = x2.to_numpy()
    y = df["death_yn"].values
    le = preprocessing.LabelEncoder()
    yLE = le.fit_transform(y)
    Xdfs.append(xOHE)
    ys.append(yLE)
    x_train, x_test, y_train, y_test = train_test_split(xOHE, yLE, test_size=0.30, random_state=123, stratify=yLE) #using stratify since its unbalanced
    covTree(x_train, x_test, y_train, y_test,str(num))
    num+=1
mDepth=[12,12,12,8,12,14]
for index in range(0,len(Xdfs)):
    print("\nTree ",index)
    x_train, x_test, y_train, y_test = train_test_split(Xdfs[index], ys[index], test_size=0.30, random_state=123, stratify=ys[index]) #using stratify since its unbalanced
    treeDepth(x_train, x_test, y_train, y_test, mDepth[index])