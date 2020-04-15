import pandas as pd 
import numpy as np 
from datetime import date, timedelta
import requests as Req 
from sklearn.preprocessing import OneHotEncoder
import tensorflow
import keras 

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K 
from keras.utils.np_utils import to_categorical 

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

#---------------------------------------
def scrape_historical_data(stock):
        '''
        takes a stock code (a string) as input and 
        returns its daily historical data for the last 5 years

        '''

        today = date.today()
        from_ = today - timedelta(days=5*365+2)

        url_first_part = "https://www.nasdaq.com/api/v1/historical/"
        url_stockCode = stock
        url_last = "/stocks/{}/{}".format(from_, today)
        my_url = url_first_part + url_stockCode + url_last 

        page_response = Req.get(my_url, allow_redirects=True, headers=headers)

        file_name = "data_folder/" + url_stockCode + ".csv"
        with open(file_name, 'wb') as f: 
            for line in page_response:
                f.write(line)


#---------------------------------------
# cadlestick() labels the daily data with an appropiriate candle name
def candlestick(Open, High, Low, Close, Volume):

    '''
    Takes the Open, High, Low, Close (Adj Close) 
    values and returns the corresponding candlestick 
    name. 

    Definitions are obtained from:
    https://www.candlesticker.com/BasicCandlesticks.aspx?lang=en

    '''
    High = float(High[2:])
    Low = float(Low[2:])
    Open = float(Open[2:])
    Close = float(Close[2:])
    back = High - Low
    if Close > Open:
        # WHITE
        if High > Close:
            if Low < Open:
                front = Close - Open 
                relative_length = front / back 
                
                if relative_length <= 0.1 :
                	return "White Spinning Top"
                elif relative_length <= 0.3 :
                	return "Short White"
                elif relative_length >= 0.8 :
                	return "Long White"
                else:
                	return "White" 
            
            else: # Low = Close
                return "White Opening Marubozu"

        else: # High = Open
            if Low < Open:
                return "White Closing Marubozu"
            else: # Low = Close
                return "White Marubozu"

    elif Open > Close:
        # BLACK
        if High > Open:
            if Low < Close:
                front = Open - Close
                relative_length = front / back
                if relative_length <= 0.1:
                    return "Black Spinning Top"
                elif relative_length <= 0.3:
                    return  "Short Black"
                elif relative_length >= 0.8:
                    return "Long Black" 
                else: 
                    return "Black"
            else: # Low = Open
                return "Black Closing Marubozu"

        else: # High = Close
            if Low < Close:
                return "Black Opening Marubozu"
            else: # Low = Open
                return "Black Marubozu"

    else: # Open = Close
        if High > Low:
            if High == Open:
                return "Umbrella"

            elif Low == Open:
                return "Inverted Umbrella"

            else: 
                return "Doji"

        else: # High = Low
            return "Four Price Doji"

#---------------------------------------
def day_trade_return(row):
    # returns the daily return rate
    buy  = float( row[' Open'][2:] )
    sell = float( row[' Close/Last'][2:] )

    return 100*(sell - buy)/buy 

#---------------------------------------

THRESHOLD = 0.5
def precision_nn(y_true, y_pred, threshold_shift=0.5-THRESHOLD):

    # just in case 
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

    precision = tp / (tp + fp)
    return precision

#---------------------------------------
def recall_nn(y_true, y_pred, threshold_shift=0.5-THRESHOLD):

    # just in case 
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    recall = tp / (tp + fn)
    return recall

#---------------------------------------
def fscore_nn(y_true, y_pred, threshold_shift=0.5-THRESHOLD):

    # just in case 
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return 2 * (precision * recall) / (precision + recall) 


#---------------------------------------
    
def nn_model(X_train, X_valid, y_train, y_valid, size=1):
    '''
    takes preprocessed X, y, and a size arguement
     - takes the first size fraction of the data
     - runs a NN model 
     - return the training and validation error
    '''
    

    X_tra = X_train[: int(round(X_train.shape[0]*size))]
    X_val = X_valid[: int(round(X_valid.shape[0]*size))]
    y_tra = y_train[: int(round(y_train.shape[0]*size))]
    y_val = y_valid[: int(round(y_valid.shape[0]*size))]

    y_tra = to_categorical(y_tra, 2)
    y_val = to_categorical(y_val, 2)
    
    [m, X_nodes] = X_tra.shape
    
    model = Sequential()
    model.add(Dense(30,activation='relu', input_shape=(X_nodes,)))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[precision_nn, recall_nn, fscore_nn, "accuracy"])
    
    # fit the model
    history = model.fit(X_tra, y_tra,  epochs=6, validation_data = (X_val, y_val))
    
    train_err = history.history['loss'][-1]
    valid_err = history.history['val_loss'][-1]
    
    return [train_err, valid_err]

#---------------------------------------
def label_data(df): 
	
	df["candle0"] = df.apply(lambda row: candlestick(row[' Open'], row[' High'], row[' Low'], row[' Close/Last'], row[' Volume'] ), axis =1)
	df["day_trade_profit_percent"] = df.apply( lambda row: day_trade_return(row), axis =1)

	# Create the features for the last four days' candle stick formations
	candle1 = df.iloc[3:-1].candle0.reset_index(drop = True)
	candle2 = df.iloc[2:-2].candle0.reset_index(drop = True)
	candle3 = df.iloc[1:-3].candle0.reset_index(drop = True)
	candle4 = df.iloc[:-4].candle0.reset_index(drop = True)

	Volume1 = df.iloc[3:-1][" Volume"].reset_index(drop = True)
	Volume2 = df.iloc[2:-2][" Volume"].reset_index(drop = True)
	Volume3 = df.iloc[1:-3][" Volume"].reset_index(drop = True)
	Volume4 = df.iloc[:-4][" Volume"].reset_index(drop = True) 

	df = df.iloc[4:].reset_index(drop = True)

	df["candle1"] = candle1
	df["candle2"] = candle2
	df["candle3"] = candle3
	df["candle4"] = candle4

	df["Volume1"] = Volume1
	df["Volume2"] = Volume2
	df["Volume3"] = Volume3
	df["Volume4"] = Volume4

	df = df.loc[ (df.Volume1 != " N/A") & (df.Volume2 != " N/A") & (df.Volume3 != " N/A") & (df.Volume4 != " N/A") ]
	df = df.astype({"Volume1":'int64', "Volume2":'int64', "Volume3":'int64', "Volume4":'int64', })

	return df

            