import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from src import constants
# Import dataset (should be 35 attributes)
def getTrainSample():
    return getFile("data/raw/sampletrain.csv")

def getTrain():
    """
    Gets Train data
    """
    return getFile("data/raw/train.csv")

def getTest():
    """
    Gets Test data
    """
    return getFile("data/raw/test.csv")

def getFile(csv):
    """
    Reads in CSV file
    """
    return pd.read_csv(csv,
        names=['ID', 'Loan Amount', 'Funded Amount', 'Funded Amount Investor', 'Term', 
            'Batch Enrolled', 'Interest Rate', 'Grade', 'Sub Grade', 'Employment Duration', 
            'Home Ownership', 'Verification Status', 'Payment Plan', 'Loan Title', 'Debit to Income', 
            'Delinquency - two years', 'Inquires - six months', 'Open Account', 'Public Record', 
            'Revolving Balance', 'Revolving Utilities', 'Total Accounts', 'Initial List Status', 
            'Total Received Interest', 'Total Received Late Fee', 'Recoveries', 'Collection Recovery Fee', 
            'Collection 12 months Medical', 'Application Type', 'Last week Pay', 'Accounts Delinquent', 
            'Total Collection Amount', 'Total Current Balance', 'Total Revolving Credit Limit', 
            'Loan Status'],
        dtype={'ID': int, 
            'Loan Amount': float, 
            'Funded Amount': float, 
            'Funded Amount Investor': float,
            'Term': int, 
            'Batch Enrolled': str,
            'Interest Rate': float,
            'Grade': str,
            'Sub Grade': str,
            'Employment Duration': str,
            'Home Ownership': float,
            'Verification Status': str,
            'Payment Plan': str,
            'Loan Title': str,
            'Debit to Income': float,
            'Delinquency - two years': int,
            'Inquires - six months': int,
            'Open Account': int,
            'Public Record': int,
            'Revolving Balance': float,
            'Revolving Utilities': float,
            'Total Accounts': int,
            'Initial List Status': str,
            'Total Received Interest': float,
            'Total Received Late Fee': float,
            'Recoveries': float,
            'Collection Recovery Fee': float,
            'Collection 12 months Medical': int,
            'Application Type': str,
            'Last week Pay': int,
            'Accounts Delinquent': int,
            'Total Collection Amount': float,
            'Total Current Balance': float,
            'Total Revolving Credit Limit': int,
            'Loan Status': bool
            },
        na_values=['-'], 
        encoding = "ISO-8859-1"
    )

def findMissingValues(df):
    """
    Finds any cells with 'NA' in it. Might need to expand to other 'missing' 
    data indicators.
    """
    missing_values = []
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            # Not sure if there are any other 'na' indicators like '-' in the dataset
            if pd.isna(df.iloc[row, col]):
                missing_values.append((row, col))
    return missing_values

def detectOutliers(df, col_name, z_thresh=3):
    """
    Takes in a data frame, column name, and z-score threshold and 
    returns a list of indexes for rows containing outliers in each given column
    
    Small note: The index this spits out will be TWO below what we see on
    the train.csv, this is because we deleted the column name info and the 
    rows starts with number '1' on excel. 
    """
    col_mean = np.mean(df[col_name])
    col_std = np.std(df[col_name])
    z_scores = abs((df[col_name] - col_mean) / col_std)
    return list(np.where(z_scores > z_thresh)[0])

def getOutlierIndexByZScore(df, colTypes, z_thresh=3):
    """
    Takes in a data frame and a dictionary of column names with their types and
    returns a dictionary of column names as keys and a list of outliers as values
    """
    colWithZScore = {}
    for key, value in colTypes.items():
       if value in [int, float]:
           colWithZScore.setdefault(key, [])
           colWithZScore[key].append(detectOutliers(df, key, z_thresh))
    return colWithZScore
           
def plot_data(df, x_axis=None):
    """
    Visually represents each attribute against the loan status attribute.
    Only purpose of this is to study the data, cause it is hard to 
    interprate without a model. 
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Create a scatter plot if data is numeric
            x_data = df[x_axis] if x_axis is not None else df.index
            plt.scatter(x_data, df[column])
            plt.xlabel(x_axis if x_axis is not None else 'Index')
            plt.ylabel(column)
            plt.show()
        else:
            # Create a histogram if data is nominal
            value_counts = df[column].value_counts()
            plt.bar(value_counts.index.astype(str), value_counts)
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.show()

def getParallelGraph(df):
    """
    Creates Parallel Graph
    """

    fig = px.parallel_coordinates(
    df, 
    color='Loan Status', 
    color_continuous_scale=px.colors.diverging.Tealrose,
    labels={
            'ID', 'Loan Amount', 'Funded Amount', 'Funded Amount Investor', 'Term', 
    'Batch Enrolled', 'Interest Rate', 'Grade', 'Sub Grade', 'Employment Duration', 
    'Home Ownership', 'Verification Status', 'Payment Plan', 'Loan Title', 'Debit to Income', 
    'Delinquency - two years', 'Inquires - six months', 'Open Account', 'Public Record', 
    'Revolving Balance', 'Revolving Utilities', 'Total Accounts', 'Initial List Status', 
    'Total Received Interest', 'Total Received Late Fee', 'Recoveries', 'Collection Recovery Fee', 
    'Collection 12 months Medical', 'Application Type', 'Last week Pay', 'Accounts Delinquent', 
    'Total Collection Amount', 'Total Current Balance', 'Total Revolving Credit Limit', 
    'Loan Status'
    },
    width=1600,
    height=1400
    )

    return fig

def getNumericalData(df):
    """
    Returns numerical dataframe from a dataframe
    """
    nums = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            nums.append(column)

    return df[nums]

def getCategoricalData(df):
    """
    Returns categorical dataframe from a dataframe + Loan status
    """
    cats = []

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            cats.append(column)
        if column == 'Loan Status':
            cats.append(column)

    return df[cats]

def pre_process_data(df):

    # Report missing values in dataset
    # if findMissingValues(df) == []: No missing values and this take up like 30 seconds so... yeah lol
    print("No missing vlaues found")

    # Remove indexes that contain 2 or more attributes with data more than 5 standard deviations away from mean
    outlier_indexes = getOutlierIndexByZScore(df, constants.columnDataTypes,5)
    mymap = {}
    list_of_indexes = []
    for key, value in outlier_indexes.items():
        for index in value[0]:
            mymap[index] = mymap.get(index, 0) + 1
            # Saves the index in the dataframe where there are 3 or more outliers
            if mymap[index] == 3:
                list_of_indexes.append(index)
                # Sets an attribute to '0' for later deletion of said row. 
                df.loc[index, 'Loan Amount'] = 0
    # Deletes the above indexes        
    df2 = df.drop(df[df['Loan Amount'] == 0].index)

    # Save two new data frames, one numerical, and one catagorical. 
    numericalDF = getNumericalData(df2)
    numericalDF.to_csv('./data/processed/numericalTrain', index=False)

    catagoricalData = getCategoricalData(df2)
    catagoricalData.to_csv('./data/processed/catagoricalTrain', index=False)