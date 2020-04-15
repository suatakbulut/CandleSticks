# Read the paper: Profitability of Candlestick Charting Patterns 
# in the Stock Exchange of Thailand
# by Piyapas Tharavanij , Vasan Siraprapasiri,and Kittichai Rajchamaha1
# at https://journals.sagepub.com/doi/pdf/10.1177/2158244017736799


# import packages
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

print("---------------------------------------------------------------")
print("\nAttention: We will be using only the last four days' candle stick formations!..\n")
print("---------------------------------------------------------------")
isScrape   = input("Do you want to skip data scraping? (y / n): ")
isLabel    = input("Do you want to skip data labeling? (y / n): ")
isEDA      = input("Do you want to skip EDA? (y / n): ")
isLogit    = input("Do you want to skip Logistic Regression? (y / n): ")
isForest   = input("Do you want to skip Random Forest Classifier? [Training a Random Forest Classifier model with this much of data might require some time and power.] (y / n): ")
isNN       = input("Do you want to skip NN? (y / n): ")
isMoreData = input("Do you want to skip checking whether the model needs more data or to be replaced? (y / n): ")

# ---------------------------------------------------------------
#     PART 1: SCRAPE DATA and CREATE VARIABLES OF INTEREST
# ---------------------------------------------------------------
# top 241 stock in SP500 based on their Market Cap
stocks = [
"MSFT", "AAPL", "AMZN", "GOOG", "GOOGL", "FB", "BRK.B", "V", "WMT", "JPM", "PG", 
"MA", "UNH", "INTC", "VZ", "T", "HD", "BAC", "MRK", "DIS", "PFE", "PEP", "CSCO", 
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
# one can skip scraping if they already have the scraped data
# isScrape = input("Do you want to scrape and label the data again? (y / n): ")
if isScrape != "y": 
    # --------------------------------
    #       Data Scraping
    # --------------------------------
    from functions import scrape_historical_data
    # scrape_historical_data : takes a stock code returns last 5 years daily data
    
    print("\nStarted scraping data. ")
    for stock in stocks: 
        scrape_historical_data(stock)    
    print("\nAll {} Stocks' historical data has downloaded. \n".format(len(stocks)))
    
    # -----------------------------------
    #  Labeling data & Creating Features
    # -----------------------------------  

if isLabel != "y": 
    print("Now, labeling the data and creating the features of interest.\n")    
    
    from functions import label_data
    # candlestick : takes a daily values comb returns candles tick formation of the day 
    # day_trade_return : % return when buy at Open and sell at Close price in a day
    
    # initialize an empty data frame in which we will be storing our data
    all_stocks = pd.DataFrame()
    
    # In order to list files in the data_folder directory import os
    import os
    
    for filename in os.listdir('data_folder'):
        # For all the stocks in our data folder, 
        # for each day, we will create the that day's candle
        # as well as the previous four days, namely candle1, ... , candle4
        # where candle i is the candle formation i day(s) ago, i=0,1,2,3,4
        data_path = 'data_folder/' + filename        
        print("Loading {}".format(filename))   
        
        if os.stat(data_path).st_size > 5:
                
            df = pd.read_csv(data_path)
            df["Stock_Code"] = filename[:-4]            
            
            df = label_data(df)
            # append this stock to our pile of data and do the same for the next stock
            all_stocks = all_stocks.append(df)
        else:
            print("Flag: {}'s size is empty.".format(filename))   
            
    
    all_stocks = all_stocks.reset_index(drop = True)
    all_stocks["isProfit"] = 0
    all_stocks.loc[all_stocks["day_trade_profit_percent"]> 0, "isProfit"] = 1
    
    all_stocks["last4days"] = all_stocks.apply(lambda row: row.candle1 + row.candle2 + row.candle3 + row.candle4, axis=1)
    # Let's save our data set to skip the scrape and label part next time
    all_stocks.to_pickle("./complicated_candles_volumes_stocks.pkl")
    print("\nall_stocks data saved as complicated_candles_volumes_stocks.pkl \n") 
else:
    print("\n------------------------------------------------------- \n")
    print("Proceeding with the data pickled in complicated_candles_volumes_stocks.pkl. \n")
    all_stocks = pd.read_pickle("./complicated_candles_volumes_stocks.pkl")
  
# ---------------------------------------------------------------
#            PART 2: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------
if isEDA != "y": 
    
    # -------------------------------
    #   2.a. Candle1 vs Return in %
    # -------------------------------
    plt.figure(figsize=(16,16))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    org_labels = all_stocks.candle1.unique()
    after_labels = ["".join([word[0] for word in label.split()]) for label in org_labels]
    g = sns.catplot(x="day_trade_profit_percent", y="candle1", kind="box", orient="h", height=4, aspect=4, whis=7, data=all_stocks)
    g.axes[0][0].axvline(0, ls='--', c="red")
    plt.title("Day-Trade Return (%) Box Plot for Different Candle Stick Formations of Yesterday.")
    plt.xlabel("Day-Trade Return Percentage")
    plt.ylabel("Yesterday's Candle Stick Formation")
    plt.show()
    
    # -------------------------------------------------
    #  2.b.  Cabdle1 & Candle2 vs Return in %
    # -------------------------------------------------
    
    plt.figure(figsize=(16,16))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    org_labels = all_stocks.candle2.unique()
    after_labels = ["".join([word[0] for word in label.split()]) for label in org_labels]
    ax = sns.catplot(x="day_trade_profit_percent", y="candle2", row="candle1",
                    kind="box", orient="h", height=4, aspect=4, whis=7,
                    data=all_stocks)
    for subgraph in ax.axes:
        subgraph[0].axvline(0, ls='--', c="red")
    ax.set_yticklabels(after_labels)
    plt.title("Boxplot of Day Trade Return (%) for Candle Formations of the Last Two Days. ")
    plt.show()
    
    # ----------------------------------
    #  2.c.  Transition Matrix and HeatMap
    # ----------------------------------
    
    # let us create a heatmap of the transition matrix
    # from yesterday's candle formation to today's
    trans = all_stocks[["candle1", "candle0"]]
    
    # since the possible candles for yesterday and today 
    # are the same I will use the same list to loop over the 
    # possible candle formations for both yesterday and today
    candles_list = list(trans.candle0.unique())
    
    # initialize the transition matrix
    candle_num = len(candles_list)
    trans_matrix = np.zeros((candle_num,candle_num))
    for cand1 in candles_list:
        for cand0 in candles_list:
            trans_matrix[candles_list.index(cand1)][candles_list.index(cand0)] = len(trans.loc[(trans.candle1 == cand1) & (trans.candle0 == cand0)])
    
    # let us create a probability transition matrix, names heatmap_df 
    # from our transition matrix trans_matrix, where the rows and 
    # columns represent the candle formation of yesterday and today's, 
    # respectively
    
    heatMap_df = pd.DataFrame(trans_matrix, columns = candles_list, index = candles_list)
    heatMap_df["row_sum"] = heatMap_df.apply(lambda row: sum(row), axis=1)
    heatMap_df = heatMap_df.apply(lambda row: row / row.row_sum, axis=1)
    heatMap_df.drop(columns=["row_sum"], inplace= True)
    heatMap_df.dropna(axis=0, inplace=True)
    
    # Since our heatmap will show the values as well, it is better 
    # if we round our probabilities to percentage points
    heatMap_rounded = round(heatMap_df, 2)
    
    
    plt.figure(figsize=(22,20))
    sns.heatmap(heatMap_rounded, annot=True, vmax = 0.3)
    plt.title("The Probability Transition Matrix From Yesterday (Row) To Today (Column)")
    plt.show()
     
# -------------------------------------------------------------------
# For the next parts, we will need to preprocess our data and create 
# our X and y variables. So I will do it here 
# -------------------------------------------------------------------
if isLogit != "y" or isForest != "y" or isNN != "y" or isMoreData != "y" : 
    
    # In the case of a Machine Learning attempt, let us split into train, validation, 
    # and test sets and preprocess our data so that we will be using the same train, 
    # valid, and test sets for different machine learning techniques 
    # at the end of this part, we will have 6 sets of data, namely:
    # X_train, X_valid, X_test, y_valid, y_train, y_test
    
    # Create X and y variables by selecting the following columns from all_stocks
    X_cols = ["candle1", "candle2", "candle3", "candle4", "Volume1", "Volume2", "Volume3", "Volume4"]
    y_col = "isProfit"
    
    X = all_stocks[X_cols] 
    y = all_stocks[y_col]
    
    # -----------------------------------
    #  Preprocessing Pipeline
    # -----------------------------------  
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Even though I have only a few columns and no such issues as missing value
    # different types of data etc, I will write down a general preprocessing
    # so that I can safely use it when I astart adding more variables into my data
    
    numerics = ["float64", "int64"]
    numerical_cols   = [col for col in X.columns if X[col].dtype in numerics ]
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    
    num_transformer = StandardScaler()
    
    cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("onehot", OneHotEncoder(handle_unknown = "ignore"))
        ])
    
    preprocessor = ColumnTransformer(
            transformers=[
                    ("num", num_transformer, numerical_cols),
                    ("cat", cat_transformer, categorical_cols )
        ])
    
    # Let us split our data into training and validation sets before even preprocessing
    # our datausing sklearn's model_selection.train_test_split package 
    # 60% train, 20% valid, and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    # Preprocess the X variable
    X_train = preprocessor.fit_transform(X_train) 
    X_valid = preprocessor.transform(X_valid) 
    X_test  = preprocessor.transform(X_test)  

# ---------------------------------------------------------------
#            PART 3: PRELIMINARY ML ATTEMPT
# ---------------------------------------------------------------
# -------------------------------
#    3.a. Logistic Regression
# -------------------------------
if isLogit != "y": 
    
    print("Employing Logistic Regression. \n")
    from sklearn.linear_model import LogisticRegression
    # Initialize LogisticRegression()
    logit_model = LogisticRegression(random_state=0, n_jobs=-1)
    
    # fit the data to the model
    logit_model.fit(X_train, y_train)
    
    # calculate the accuracy of our model
    score = logit_model.score(X_test, y_test)
    
    """
    Accuracy may not be a complete measure for our analysis. 
    Similar to spam filterings, we may be interested in the 
    recall and the precision of our algorithm, too. 
    """
    
    y_pred_logit = logit_model.predict(X_test) 
    
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_logit, pos_label = 1, average="binary")
    print("\n------------------------------------------------------------------------")
    print("\t\tLogistic Regression Resulsts")
    print("\t\t----------------------------\n")
    print("Accuracy {} \t|\t Precision {} \t|\t Recall {} \n".format(round(score,3), round(precision,3), round(recall,3)))
    print("------------------------------------------------------------------------\n")
 
# -------------------------------
#    3.b. Random Forest Classifier
# -------------------------------
if isForest != "y": 
    
    ### I need to find a way to make Random Forest Model to run more efficiently    
    print("Employing Random Forest Classifier. \n")
    from sklearn.ensemble import RandomForestClassifier
    # Initialize RandomForestClassifier()
    forest_model = RandomForestClassifier(n_estimators = 3000, n_jobs=-1)
    
   # fit the data to the model
    print("Fitting data to Random Forest Model. \n")
    forest_model.fit(X_train, y_train)
    print("Data training is complete. \n")
    
    # calculate the accuracy, precision, and recall:
    y_pred_forest = forest_model.predict(X_test)
    forest_acc = sum(y_pred_forest == y_test) / len(y_test)  
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_forest, pos_label = 1, average="binary")
    print("\n------------------------------------------------------------------------")
    print("\t\tRandom Forest Classifier Results")
    print("\t\t--------------------------------\n")
    print("Accuracy {} \t|\t Precision {} \t|\t Recall {} \n".format(round(forest_acc,3), round(precision,3), round(recall,3)))
    print("------------------------------------------------------------------------\n")

# ---------------------------------------------------------------
#            PART 4: Neural Network ATTEMPT
# ---------------------------------------------------------------
if isNN != "y": 
    
    from sklearn.preprocessing import OneHotEncoder
    import tensorflow
    import keras 
    from keras.utils.np_utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense
    import numpy as np
    from keras import backend as K
    from functions import precision_nn, recall_nn, fscore_nn
    
    # Create the number of nodes in the input layer
    X_nodes = X_train.shape[1] 
    
    # convert y_train and y_valid into vectors of 2 dims
    y_train_nn = to_categorical(y_train, 2)
    y_valid_nn = to_categorical(y_valid, 2)
    
    model = Sequential()
    model.add(Dense(50,activation='relu', input_shape=(X_nodes,)))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(2, activation='sigmoid'))   
    
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[precision_nn, recall_nn, fscore_nn, "accuracy"])
    model.summary()
    
    # fit the model
    history = model.fit(X_train, y_train_nn,  epochs=10, validation_data = (X_valid, y_valid_nn))
    
    # predict
    y_pred_float = model.predict(X_test)
    y_pred_nn = np.argmax(y_pred_float, axis=1)
    
    # evaluate the model    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_nn, pos_label = 1, average="binary")
    accuracy_nn = sum(y_test == y_pred_nn) / len(y_test)
    print("\n------------------------------------------------------------------------")
    print("\t\tNerual Network Results")
    print("\t\t---------------------- \n")
    print("Accuracy {} \t|\t Precision {} \t|\t Recall {} \n".format(round(accuracy_nn,3), round(precision,3), round(recall,3)))
    print("------------------------------------------------------------------------\n")
    
# ---------------------------------------------------------------
#            PART 5: More data, More features, or new model ?
# ---------------------------------------------------------------   
# isMoreData = input("Do you want to check whether the model needs more data or to be replaced? (y / n): ")
if isMoreData != "y": 
    # In this part we will plot training error and validation error 
    # as the training size increases in order to understand whether 
    # we have a high bias model or a high variance model
    from sklearn.model_selection import train_test_split
    from functions import nn_model
    
    train = []
    valid = []
    size_range = np.arange(0.05, 0.450, 0.005)
    for size in size_range:
        print("---------------------------------------------------")
        print("\nRunning the model using the first {%f2.2} % of the data. \n".format(size*100))
        print("---------------------------------------------------")
        [train_err, valid_err] = nn_model(X_train, X_valid, y_train, y_valid, size=size)
        train.append(train_err)
        valid.append(valid_err)
    
    [train_err, valid_err] = nn_model(X_train, X_valid, y_train, y_valid, size=1)
    train.append(train_err)
    valid.append(valid_err)
    
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,10))
    plt.plot(size_range, valid, 'red', label = "Validation_Err")
    plt.plot(size_range, train, 'blue', label = "Train_Err")
    plt.legend()
    plt.title("Change in Validation and Training Error against training size")
    plt.xlabel("% of the Data")
    plt.show()
