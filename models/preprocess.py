
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

sk_prefix = 'sagemaker/lstm_models/'

def preprocess(ticker, months=48, short_span=8, long_span=20):
    '''
    preprocess(ticker, months, short_span, long_span) returns a train and test split of the 
    desired stock's data with the desired features. ticker is a type Str of the stock ticker, 
    months is an Int value representing the number of months of data we want to train on, 
    short_span is the span of our shorter EMAs, long_span is the span for longer EMA calculations.

    Returns a dictionary of x_train, y_train, x_test, y_test, x_train_date, x_test_date, scaler.
    '''

    hist = yf.Ticker(ticker.upper()).history(period=f'{months}mo').iloc[:,:5]
    hist = ema_calc(hist, short_span, long_span)

    features = ['Open', 'High', 'Low', 'Close', 'EMA_8', 'EMA_20']

    print(hist)

    # Scale the selected features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(hist[features].values)  # Scale all relevant features
    pred_days = 30
    train_split = 0.8

    # Prepare the date index (for tracking predictions)
    date_index = hist[pred_days:].index

    x_train = []
    y_train = []

    # Create sequences of the past pred_days for training
    for x in range(pred_days, len(scaled)):
        x_train.append(scaled[x-pred_days:x])  # Include all features in the sequence
        y_train.append(scaled[x, 3])  # Predict the closing price (index 3 corresponds to 'Close')

    # # Convert to numpy arrays for training
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Split the data into training and testing sets
    split_idx = int(np.ceil(len(x_train) * train_split))
    x_train, x_test = x_train[:split_idx], x_train[split_idx:]
    y_train, y_test = y_train[:split_idx], y_train[split_idx:]

    # Reshape the data to fit the model's input requirements
    # The shape will be (number of samples, number of time steps, number of features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(features)))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(features)))

    # Optional: Separate the date index for the training and testing sets
    x_train_date, x_test_date = np.array(date_index[:split_idx]), np.array(date_index[split_idx:])

    data = {'x_train': x_train, 'y_train':y_train, 'x_test':x_test, 'y_test':y_test, 
            'x_train_date':x_train_date, 'x_test_date':x_test_date, 'scaler':scaler}

    return data

def ema_calc(df, short_span, long_span):
    '''
    ema_calc(df) takes a stock's df as input and returns the df with the 8 and 20-day EMAs
    added as their own columns
    '''

    # Calculate the Simple Moving Averages (SMA) for the initial EMA values
    df['SMA_8'] = df['Close'].rolling(window=short_span).mean()
    df['SMA_20'] = df['Close'].rolling(window=long_span).mean()

    # Calculate the 8-day EMA starting from the 8-day SMA
    df['EMA_8'] = df['Close'].ewm(span=short_span, adjust=False, min_periods=short_span).mean()

    # Calculate the 20-day EMA starting from the 20-day SMA
    df['EMA_20'] = df['Close'].ewm(span=long_span, adjust=False, min_periods=long_span).mean()

    df.drop(columns=['SMA_8', 'SMA_20'], inplace=True)
    df.dropna(inplace=True)
    return df

def graph_data(scaler, x_test, y_test, x_test_date, model):
    '''
    graph_data(scaler, x_test, y_test, x_test_date, model) takes the scaler, test data, and model, then
    returns a dictionary of the test prediction, actual test values, the x/date values, and the scaler
    for the output/y-values.
    '''
    
    y_t = y_test.reshape(-1,1)

    # Inverse transform only the Close feature
    scaler_close = StandardScaler()
    scaler_close.mean_ = scaler.mean_[3]  # Mean of the 'Close' feature
    scaler_close.scale_ = scaler.scale_[3]  # Scale of the 'Close' feature

    # Apply inverse_transform to get the actual and predicted Close prices
    y_actual = scaler_close.inverse_transform(y_t)
    y_pred = scaler_close.inverse_transform(model.predict(x_test))#.reshape(-1, 1))

    # y_actual = scaler.inverse_transform(y_t)
    # y_pred = scaler.inverse_transform(model.predict(x_test))
    new_data = np.array([x.date() for x in x_test_date])
    data = {'y_pred': y_pred, 'y_actual':y_actual, 'new_data':new_data, 'scaler_close':scaler_close}

    return data

def prediction(pred_data, scaler_close, model):
    '''
    prediction(pred_data, scaler_close, model) takes scaled input data, the respective scaler
    used on that data, and the model trained on that data, and returns the prediction based on the
    provided data. pred_data is of type 2d np.array, scaler close could be a scalar from sklearn,
    and model is any tensorflow model with the predict option.
    '''

    real_data = np.reshape(pred_data, (1, pred_data.shape[0], pred_data.shape[1]))
    prediction = model.predict(real_data)
    prediction = scaler_close.inverse_transform(prediction)
    return prediction[0]

    