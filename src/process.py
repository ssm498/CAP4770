import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import plotly.express as px
from src import pre_process
from src import constants

def getTrainAndTestSet(df):
    train_nums = pre_process.getNumericalData(df)
    train_cats = pre_process.getCategoricalData(df)

    # Drop nonuseful columns
    train_nums.drop(columns=['ID'], inplace=True)
    train_cats.drop(columns=['Batch Enrolled'], inplace=True)

    # Testing removing extra cols
    train_cats.drop(columns=['Loan Title'], inplace=True)
    train_cats.drop(columns=['Grade'], inplace=True)
    train_cats.drop(columns=['Sub Grade'], inplace=True)

    cats = pd.get_dummies(train_cats, columns=[
        #'Grade', 
        #'Sub Grade', 
        'Employment Duration', 
        'Verification Status',
        'Payment Plan',
        #'Loan Title',
        'Initial List Status',
        'Application Type'
        ], drop_first=True)

    # Concat nums and cats
    concatDf = pd.concat([train_nums, cats], axis=1, join='inner')

    train_set, test_set = train_test_split(concatDf, test_size=0.2)

    return (train_set, test_set)

def testRandomForest(train_set, test_set):
    rf = RandomForestClassifier(class_weight = 'balanced')

    train_x = train_set.drop(columns=['Loan Status'])
    train_y = train_set['Loan Status']

    test_x = test_set.drop(columns=['Loan Status'])
    test_y = test_set['Loan Status']

    model = rf.fit(train_x, train_y)

    # Predictions
    preds = rf.predict(test_x)
    
    print("Accuracy: ", accuracy_score(test_y,preds))
    print("F1 Score: ", f1_score(test_y, preds, zero_division=1))
    print(confusion_matrix(test_y,preds))

    features = pd.DataFrame({'Feature':train_x.columns, 'Importance': model.feature_importances_})

    fig = px.bar(features, x='Feature', y='Importance')
    fig.show()

def testLogisticRegression(train_set, test_set):
    lr = LogisticRegression(max_iter=5000, class_weight = 'balanced')

    train_x = train_set.drop(columns=['Loan Status'])
    train_y = train_set['Loan Status']

    test_x = test_set.drop(columns=['Loan Status'])
    test_y = test_set['Loan Status']

    model = lr.fit(train_x, train_y)

    # Predictions
    preds = lr.predict(test_x)

    print("Accuracy: ", accuracy_score(test_y,preds))
    print("F1 Score: ", f1_score(test_y, preds, zero_division=1))
    print(confusion_matrix(test_y,preds))

    features = pd.DataFrame({'Feature':train_x.columns, 'Importance': model.coef_[0]})

    fig = px.bar(features, x='Feature', y='Importance')
    fig.show()

def testKNeighborsClassifier(train_set, test_set):
    kn = KNeighborsClassifier()

    train_x = train_set.drop(columns=['Loan Status'])
    train_y = train_set['Loan Status']

    test_x = test_set.drop(columns=['Loan Status'])
    test_y = test_set['Loan Status']

    model = kn.fit(train_x, train_y)

    # Predictions
    preds = kn.predict(test_x)

    print("Accuracy: ", accuracy_score(test_y,preds))
    print("F1 Score: ", f1_score(test_y, preds, zero_division=1))
    print(confusion_matrix(test_y,preds))

    #KNN Does not have features that can be measured