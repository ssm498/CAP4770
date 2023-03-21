import pandas as pd
import numpy as np
# Import dataset (should be 35 attributes)
def getTrain():
       return getFile("data/raw/train.csv")

def getTest():
       return getFile("data/raw/test.csv")

def getFile(csv):
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
                     encoding = "ISO-8859-1")

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
    """
    col_mean = np.mean(df[col_name])
    col_std = np.std(df[col_name])
    z_scores = abs((df[col_name] - col_mean) / col_std)
    return list(np.where(z_scores > z_thresh)[0])

def getZScores(df, colTypes):
    """
    Takes in a data frame and a dictionary of column names with their types and
    returns a dictionary of column names as keys and a list of outliers as values
    """
    colWithZScore = {}
    for key, value in colTypes.items():
       if value in [int, float]:
           colWithZScore.setdefault(key, [])
           colWithZScore[key].append(detectOutliers(df, key))
    return colWithZScore
           