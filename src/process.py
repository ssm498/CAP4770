import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from src import pre_process
from src import constants

def getTrainAndTestSet(df):
    train_nums = pre_process.getNumericalData(df)
    train_cats = pre_process.getCategoricalData(df)

    # Drop nonuseful id, batch
    train_nums.drop(columns=['ID'], inplace=True)
    train_cats.drop(columns=['Batch Enrolled'], inplace=True)

    df_oneHot = OneHotEncoder().fit_transform(train_cats)

    df_cat_encoded = pd.DataFrame(df_oneHot.toarray())

    df_cat_encoded.columns = constants.categories

    # Concat nums and cats
    concatDf = pd.concat([train_nums, df_cat_encoded], axis=1, join='inner')

    train_set, test_set = train_test_split(concatDf, test_size=0.2, random_state=10)

    return (train_set, test_set)

def testRandomForest(train_set, test_set):
    # Random Forest
    rf = RandomForestClassifier()

    train_x = train_set.drop(columns=['Loan Status'])
    train_y = train_set['Loan Status']

    test_x = test_set.drop(columns=['Loan Status'])
    test_y = test_set['Loan Status']

    rf.fit(train_x, train_y)

    # Predictions
    preds = rf.predict(test_x)
    
    print("Accuracy: ", accuracy_score(preds,test_y))
    print(confusion_matrix(test_y,preds))

def testLogisticRegression(train_set, test_set):
    # Random Forest
    lr = LogisticRegression(solver='lbfgs', max_iter=5000)

    train_x = train_set.drop(columns=['Loan Status'])
    train_y = train_set['Loan Status']

    test_x = test_set.drop(columns=['Loan Status'])
    test_y = test_set['Loan Status']

    lr.fit(train_x, train_y)

    # Predictions
    preds = lr.predict(test_x)

    print("Accuracy: ", accuracy_score(preds,test_y))
    print(confusion_matrix(test_y,preds))