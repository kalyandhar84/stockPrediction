'''
References:
Table: https://physionet.org/content/gait-maturation-db/1.0.0/data/table.csv
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html
https://joblib.readthedocs.io/en/latest/why.html
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
https://towardsdatascience.com/using-joblib-to-speed-up-your-python-pipelines-dd97440c653d
https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4
'''
import joblib
from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime as dt
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
from keras.models import load_model
#import pickle as pk
from keras.models import model_from_json

app = Flask(__name__)
app.debug = True

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=80)

@app.route('/')
def homepage():
    return render_template("main.html")




@app.route('/cpp/', methods=['GET', 'POST'])
def cpp():
    t=""
    my_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    req_type = request.method
    if req_type == 'GET':
        if os.path.exists(my_path + '\\static\\images\\stock_summary.jpg'):
            os.remove(my_path + '\\static\\images\\stock_summary.jpg')
        if os.path.exists(my_path + '\\static\\images\\stock_forecast.jpg'):
            os.remove(my_path + '\\static\\images\\stock_forecast.jpg')
        return render_template("cpp.html")
    else:
        symbol = request.form['stock'] 
        t = symbol
        fb_stock_model, fb_stock_future, fb_stock_1yr = get_data(symbol)
        if request.form['submit_button'] == 'Summary':
            if os.path.exists(my_path + '\\static\\images\\stock_forecast.jpg'):
                os.remove(my_path + '\\static\\images\\stock_forecast.jpg')
            fb_52max, fb_52min, fb_90max, fb_90min = get_summary(fb_stock_1yr, symbol)
            # summary_df = pd.DataFrame()
            # summary_df = [['52 Week High', np.round(fb_52max.values)], ['52 WeekLow', np.round(fb_52min.values)],
            #  ['90 Day High', np.round(fb_90max.values)], ['90 Day Low', np.round(fb_90min.values)]]

            ##print((fb_52max, fb_52min, fb_90max, fb_90min))
        else:
            if os.path.exists(my_path + '\\static\\images\\stock_summary.jpg'):
                os.remove(my_path + '\\static\\images\\stock_summary.jpg')
            predict = ml_modelprocess(fb_stock_model, fb_stock_future, symbol)
    return render_template("cpp.html", p = t)




'''
##############################################################
JUGAL CODE START HERE
#############################################################

'''
'''
1) Get Stock Symbol
2) Read Stock Data using get_data function
3) If forecast then Run ML model Process else get_summary





'''


def stock_reader(symbol):
    start_date = '2015-01-01'

    tmp_df = yf.download(symbol, start=start_date, end=dt.today())[['Adj Close']]
    tmp_df.columns = ['close_' + symbol]
    tmp_df['close_' + symbol] = pd.to_numeric(tmp_df['close_' + symbol])
    tmp_df = tmp_df.sort_index()
    tmp_df.index.name = 'Date'
    return tmp_df


''' Below two functions are added / changed. Please delete load_model function
from old file and add below two functions

'''


def load_models(symbol):
    symbol = symbol.lower()
    scaler_file = symbol.lower() + "_scaler.pkl"
    # model_file = symbol.lower()+"_lstm.pkl"
    # print(scaler_file)
    scaler_model = joblib.load(scaler_file)
    lstm_model = load_keras_model(symbol)

    return scaler_model, lstm_model


def load_keras_model(symbol):
    symbol = symbol.lower()
    # load json and create model
    json_file = open(symbol + "_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(symbol + "_model.h5")
    print("Loaded model from disk")

    return loaded_model


''' Keep code as it is '''


def get_symbol():
    text = 'FB'
    return text


def get_data(symbol):
    # np_arr = float_str_np_array(text)
    # np_arr = np_arr.astype(int)
    # np_arr[0] = np_arr[0] - 1900
    # print(np_arr)
    # scaler = StandardScaler()
    yr1 = dt.now() - datetime.timedelta(days=1 * 365)
    df_stock_all = stock_reader(symbol)
    df_stock_model = df_stock_all[:'2022-02-28']
    df_stock_future = df_stock_all['2022-03-01':]
    df_stock_1yr = df_stock_all[yr1:]

    df_stock_future = pd.concat([df_stock_model.tail(5), df_stock_future])
    return df_stock_model, df_stock_future, df_stock_1yr


def get_future_X(df_future, scaler_model):
    future_values = df_future.values
    future_values_scaled = scaler_model.transform(future_values)
    return future_values_scaled


def preprocessing_future(dataset, time_steps=1):
    X = []
    for i in range(len(dataset) - time_steps - 1):
        x = dataset[i:(i + time_steps), 0]
        X.append(x)
        # y.append(dataset[i + time_steps, 0])
    return np.array(X)


def get_summary(df_1yr, symbol):
    day_90 = dt.now() - datetime.timedelta(days=90)
    df_90day = df_1yr[day_90:]

    df_rolling7 = df_1yr.rolling(window=7).mean().dropna()
    df_rolling30 = df_1yr.rolling(window=30).mean().dropna()

    df_52max = df_1yr.max()
    df_52min = df_1yr.min()

    df_90max = df_90day.max()
    df_90min = df_90day.min()

    plt.figure(figsize=(14, 8))
    plt.plot(df_1yr, 'black', linewidth=3, label='Actual')
    plt.plot(df_rolling7, 'r--', label='7 day moving Average')
    plt.plot(df_rolling30, 'b--', label='30 day moving Average')
    plt.legend()
    plt.ylabel('Stock Price', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.title("Stock Price Movement - "+symbol, fontsize=20)
    my_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Figures out the absolute path for you in case your working directory moves around.

    plt.savefig(my_path + '\\static\\images\\stock_summary.jpg')

    return df_52max, df_52min, df_90max, df_90min


''' Below fucntion is slighlty modified. Please delete old ml_modelprocess
function and update it with new one. '''


def ml_modelprocess(df_model, df_future, symbol):
    scaler_model, lstm_model = load_models(symbol)
    X_future = get_future_X(df_future, scaler_model)
    X_future = preprocessing_future(X_future, 5).reshape(-1, 5, 1)
    y_future = lstm_model.predict(X_future)
    y_future_prices = scaler_model.inverse_transform(y_future)

    predict_df = pd.DataFrame()
    # print(df_future.index[5:-1])
    predict_df['Date'] = df_future.index[5:-1]
    predict_df.set_index('Date', inplace=True)

    predict_df['close_' + symbol] = y_future_prices
    predict_df = pd.concat([df_model.tail(1), predict_df])

    plt.figure(figsize=(14, 8))
    plt.plot(df_model.tail(300), 'b', label='Actual')
    plt.plot(predict_df, 'r--', label='Forecast')
    plt.legend()
    plt.ylabel('Stock Price', fontsize=13)
    plt.xlabel('Date', fontsize=13)
    plt.title(symbol + " Stock Price Movement with Forecast", fontsize=20)
    my_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plt.savefig(my_path + '\\static\\images\\stock_forecast.jpg')


def float_str_np_array(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)
