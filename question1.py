import requests as Req
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------
#            PART 1: SCRAPE DATA
# ---------------------------------------------------------------

def scrape_historical_data(stock):
    '''
    takes a string stock code as input and 
    returns its daily historical data from 
    2015-03-31 to 2020-03-31
    '''
    first = "https://www.nasdaq.com/api/v1/historical/"
    stockCode = stock
    last ="/stocks/2015-03-31/2020-03-31"
    my_url = first + stockCode + last 

    page_url = Req.get(my_url)

    file_name = "data_folder/" + stockCode + ".csv"
    with open(file_name, 'wb') as f: 
        for line in page_url:
            f.write(line)

# '''
# # top 100 stock in SP500 based on their Market Cap
# stocks = [
# "MSFT", "AAPL", "AMZN", "GOOG", "GOOGL", "FB", "BRK.B", "JNJ", "V", "WMT", 
# "JPM", "PG", "MA", "UNH", "INTC", "VZ", "T", "HD", "BAC", "KO", 
# "MRK", "DIS", "PFE", "PEP", "CSCO", "CMCSA", "ORCL", "NFLX", "XOM", "NVDA", 
# "ADBE", "ABT", "CRM", "NKE", "CVX", "LLY", "COST", "WFC", "MCD", "MDT", 
# "BMY", "AMGN", "NEE", "PYPL", "TMO", "PM", "ABBV", "ACN", "CHTR", "LMT", 
# "AMT", "DHR", "UNP", "IBM", "TXN", "HON", "AVGO", "GILD", "C", "BA", 
# "LIN", "UTX", "UPS", "SBUX", "MMM", "CVS", "QCOM", "FIS", "AXP", "TMUS", 
# "MDLZ", "MO", "BLK", "LOW", "GE", "FISV", "CME", "D", "CI", "INTU", 
# "SYK", "SO", "DUK", "BDX", "PLD", "CAT", "EL", "SPGI", "ISRG", "CCI", 
# "AGN", "TJX", "ADP", "VRTX", "ANTM", "CL", "GS", "AMD", "USB", "ZTS", 
# ]
# '''

# top 250 stock in SP500 based on their Market Cap
stocks = [
"MSFT", "AAPL", "AMZN", "GOOG", "GOOGL", "FB", "BRK.B", "JNJ", "V", "WMT", "JPM", "PG", 
"MA", "UNH", "INTC", "VZ", "T", "HD", "BAC", "KO", "MRK", "DIS", "PFE", "PEP", "CSCO", 
"CMCSA", "ORCL", "NFLX", "XOM", "NVDA", "ADBE", "ABT", "CRM", "NKE", "CVX", "LLY", "COST", 
"WFC", "MCD", "MDT", "BMY", "AMGN", "NEE", "PYPL", "TMO", "PM", "ABBV", "ACN", "CHTR", 
"LMT", "DHR", "UNP", "IBM", "TXN", "HON", "AVGO", "GILD", "C", "BA", "LIN", "UTX", 
"UPS", "SBUX", "MMM", "CVS", "QCOM", "FIS", "AXP", "TMUS", "MDLZ", "MO", "BLK", "LOW", "GE", 
"FISV", "CME", "D", "CI", "INTU", "SYK", "SO", "BDX", "PLD", "CAT", "EL", "SPGI", 
"ISRG", "CCI", "AGN", "TJX", "ADP", "VRTX", "ANTM", "CL", "GS", "AMD", "USB", "ZTS", "NOC", 
"MS", "NOW", "BIIB", "BKNG", "EQIX", "REGN", "CB", "MU", "TGT", "ITW", "ECL", "TFC", 
"ATVI", "CSX", "GPN", "SCHW", "MMC", "PGR", "PNC", "BSX", "KMB", "APD", "DE", "SHW", "AMAT", 
"AEP", "MCO", "EW", "WM", "BAX", "LHX", "NSC", "ILMN", "RTN", "HUM", "WBA", "SPG",  
"GD", "NEM", "DG", "SRE", "LRCX", "EXC", "DLR", "PSA", "ADI", "ROP", "CNC", "LVS", "COP", 
"FDX", "GIS", "KMI", "ADSK", "XEL", "ETN", "GM", "MNST", "ROST", "KHC", "HCA", "SBAC", "BK", 
"MET", "WEC", "ALL", "EMR", "STZ", "EA", "HSY", "ES", "ED", "SYY", "CTSH", "AFL", 
"MAR", "TRV", "COF", "DD", "HRL", "HPQ", "RSG", "EBAY", "INFO", "MSCI", "EQR", "ORLY", "MSI", 
"TROW", "KR", "PSX", "VFC", "AVB", "PEG", "VRSK", "KLAC", "AIG", "MCK", "APH", "A", "AWK", 
"CLX", "PAYX", "WLTW", "DOW", "PRU", "TEL", "BLL", "EOG", "FE", "IQV", "YUM", "PCAR", "F", 
"RMD", "WELL", "K", "VRSN", "EIX", "PPG", "AZO", "JCI", "TWTR", "CMI", "IDXX", "TT", "ZBH", 
"O", "PPL", "ETR", "HLT", "ANSS", "SLB", "DAL", "CTAS", "LUV", "DTE", "XLNX", "SNPS", 
"ADM", "ALXN", "VLO", "AEE", "CERN", "DLTR"
]

isScrape = input("Do you want to scrape the data again? (y / n)")
if isScrape == "y":
    for stock in stocks:
        scrape_historical_data(stock)

print("All 250 Stocks' historical data has downloaded. ")


# ---------------------------------------------------------------
#            PART 2: PREPARE VARIABLES OF INTEREST
# ---------------------------------------------------------------


def candlestick(Open, High, Low, Close, Volume):

    '''
    Takes the Open, High, Low, Close (Adj Close) 
    values and returns the corresponding candlestick 
    name. 

    Definitions are obtained from:
    https://www.candlesticker.com/BasicCandlesticks.aspx?lang=en

    '''

    if Close > Open:
        # WHITE
        if High > Close:
            if Low < Open:
                return "W" 
            else: # Low = Close
                return "WOM"

        else: # High = Open
            if Low < Open:
                return "WCM"
            else: # Low = Close
                return "WM"

    elif Open > Close:
        # BLACK
        if High > Open:
            if Low < Close:
                return "B"
            else: # Low = Open
                return "BCM"

        else: # High = Close
            if Low < Close:
                return "BOM"
            else: # Low = Open
                return "BM"

    else: # Open = Close
        if High > Low:
            if High == Open:
                return "U"

            elif Low == Open:
                return "IU"

            else: 
                return "D"

        else: # High = Low
            return "FPD"

def day_trade_prof(row):
    buy  = float( row[' Open'][2:] )
    sell = float( row[' Close/Last'][2:] )

    return 100*(sell - buy)/buy 

all_stocks = pd.DataFrame()

for filename in os.listdir('data_folder'):
    print("Loading {}".format(filename))        
    data_path = 'data_folder/' + filename
    df = pd.read_csv(data_path)
    df["Stock_Code"] = filename[:-4]
    df["candle0"] = df.apply(lambda row: candlestick(row[' Open'], row[' High'], row[' Low'], row[' Close/Last'], row[' Volume'] ), axis =1)
    df["day_trade_profit_percent"] = df.apply( lambda row: day_trade_prof(row), axis =1)
    
    candle1 = df.iloc[3:-1].candle0.reset_index(drop = True)
    candle2 = df.iloc[2:-2].candle0.reset_index(drop = True)
    candle3 = df.iloc[1:-3].candle0.reset_index(drop = True)
    candle4 = df.iloc[:-4].candle0.reset_index(drop = True)

    df = df.iloc[4:].reset_index(drop = True)

    df["candle1"] = candle1
    df["candle2"] = candle2
    df["candle3"] = candle3
    df["candle4"] = candle4

    all_stocks = all_stocks.append(df)

all_stocks = all_stocks.reset_index(drop = True)
all_stocks["isProfit"] = 0
all_stocks.loc[all_stocks["day_trade_profit_percent"]>= 1, "isProfit"] = 1
cols = [ "candle1", "candle2", "candle3", "candle4", "isProfit"]
X = all_stocks[cols]


X["last4days"] = X.apply(lambda row: row.candle1 + row.candle2 + row.candle3 + row.candle4, axis=1)

# ---------------------------------------------------------------
#            PART 3: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------

from matplotlib import pyplot as plt
import seaborn as sns

for candle in ["candle1", "candle2", "candle3", "candle4"]:
    plt.figure(figsize = (8,8))
    sns.catplot(x=candle, y="isProfit", kind="bar", data=X)  
    plt.title("Candle Formation in {} Day(s) Ago vs Average Profit Today".format(candle[-1]))
    plt.show()

plt.figure(figsize = (8,15))
sns.catplot(x="last4days", y="isProfit", kind="bar", data=X)
plt.title("Candle Formation Sequence (Last 4 days) vs Average Profit Today".format(candle[-1]))
plt.show()


# ---------------------------------------------------------------
#            PART 4: PRELIMINARY ML ATTEMPT
# ---------------------------------------------------------------
print("Preliminary Machine Learning Attempt to our data. \n")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Even though I have only a few columns and no such issues as missing value
# different types of data etc, I will write down a general preprocessing
# so that I can safely use it when I astart adding more variables into my data

y = X.isProfit
X = X.drop(['isProfit'], axis=1)

numerics = ["float64", "int64"]
numerical_cols   = [col for col in X.columns if X[col].dtype in numerics ]
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

num_transformer = SimpleImputer(strategy="mean")
cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown = "ignore"))
    ])

preprocessor = ColumnTransformer(
        transformers=[
                ("num", num_transformer, numerical_cols),
                ("cat", cat_transformer, categorical_cols )
    ])



# Let us split our data into training and validation sets using sklearn's
# model_selection.train_test_split package 
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
print("Data is split into train and validation sets. \n")

# -----------------------------------------------------------------------------
#       4.a. Logistic Regression
# -----------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

logit_model = LogisticRegression(random_state=0, n_jobs=-1)

logit_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("model", logit_model )
    ])

logit_pipeline.fit(X_train, y_train)
print("Fitting the data to our logit model. \n")
score = logit_pipeline.score(X_valid, y_valid)

y_pred = logit_pipeline.predict(X_valid)
# calculate the accuracy


"""
Accuracy may not be a complete measure for our analysis. 
Similar to spam filterings, we may be interested in the 
recall and the precision of our algorithm, too. 

"""

from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(y_valid, y_pred, pos_label = 1, average="binary")
print("Accuracy {} \t|\t Precision {} \t|\t Recall {} \n".format(score, precision, recall))