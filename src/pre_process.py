import pandas as pd

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
                        'Debt to Income': float,
                        'Delinquency - two years': int,
                        'Inquiries - six months': int,
                        'Open Accounts': int,
                        'Public Records': int,
                        'Revolving Balance': float,
                        'Revolving Line Utilization': float,
                        'Total Accounts': int,
                        'Initial List Status': str,
                        'Total Interest Received': float,
                        'Total Late Fees Received': float,
                        'Recoveries': float,
                        'Collection Recovery Fee': float,
                        'Collections_12_months_ex_med': int,
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