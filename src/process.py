import pandas as pd
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from statistics import mean
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
import plotly.express as px
from src import pre_process
from src import constants

def encodeDf(df):
    train_nums = pre_process.getNumericalData(df)
    train_cats = pre_process.getCategoricalData(df)

    # Drop nonuseful columns
    train_nums.drop(columns=['Accounts Delinquent'], inplace=True)
    train_nums.drop(columns=['ID'], inplace=True)
    train_cats.drop(columns=['Batch Enrolled'], inplace=True)
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

    return concatDf

def encodeDfAlternate(df):
    encodedDf = encodeDf(df)

    return encodedDf[["Loan Amount","Debt to Income","Verification Status_Source Verified","Loan Status"]]

def getTrainAndTestSet(df):
    encodedDf = encodeDf(df)

    train_set, test_set = train_test_split(encodedDf, test_size=0.2)

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
    kn = KNeighborsClassifier(leaf_size=60, weights='distance')

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

def getTrainAndTestSetWithOverSampling(df):
    encodedDf = encodeDf(df)

    # Separate the target variable from the features
    X = encodedDf.drop(columns=['Loan Status'])
    y = encodedDf['Loan Status']

    # Apply random oversampling
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Split the resampled data into train and test sets
    train_x, test_x, train_y, test_y = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Create the train and test sets as dataframes
    train_set = pd.concat([train_x, train_y], axis=1)
    test_set = pd.concat([test_x, test_y], axis=1)

    return (train_set, test_set)

def runStratifiedModel(df, type, eval = "t"):
    # Tuned using Stratified KFold and oversampling minority class
    encodedDf = None

    if eval == "t":
        encodedDf = encodeDf(df)
    elif eval == "nt":
        encodedDf = encodeDfAlternate(df)
    else:
        print("need eval")
        return

    oversample = SMOTE()

    X = encodedDf.drop(columns=['Loan Status'])
    y = encodedDf['Loan Status']
    
    xOver, yOver = oversample.fit_resample(X, y)
    xOverTrain, xOverTest, yOverTrain, yOverTest = train_test_split(xOver, yOver, test_size=0.2, stratify=yOver)
    
    model = None

    if type == "rfc":
        model = RandomForestClassifier(n_estimators=100, random_state=0)
    elif type == "knn":
        model = KNeighborsClassifier(leaf_size=60, weights='distance')
    elif type == "lr":
        model = LogisticRegression(max_iter=5000, class_weight='balanced')
    else:
        print("need type")
        return
    
    kFoldCV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

    scoring = ('f1', 'recall', 'precision')

    scores = cross_validate(model, xOver, yOver, scoring=scoring, cv=kFoldCV)

    print('Mean f1: ', mean(scores['test_f1']))
    print('Mean recall: ', mean(scores['test_recall']))
    print('Mean precision: ', mean(scores['test_precision']))

    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, stratify=y)
    
    model.fit(xOverTrain, yOverTrain)
    preds = model.predict(testX)

    #Create models metrics
    print("Accuracy: ", accuracy_score(testY, preds))
    print("F1 Score: ", f1_score(testY, preds, zero_division=1))
    print(confusion_matrix(testY, preds))

    if type == "rfc":
        features = pd.DataFrame({'Feature': trainX.columns, 'Importance': model.feature_importances_})

        fig = px.bar(features, x='Feature', y='Importance')
        fig.show()
    elif type == "knn":
        #No features to model
        print()
    elif type == "lr":
        features = pd.DataFrame({'Feature': trainX.columns, 'Importance': model.coef_[0]})

        fig = px.bar(features, x='Feature', y='Importance')
        fig.show()