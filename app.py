import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import threading
import webbrowser
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__)

# Ensure 'static' folder exists
os.makedirs("static", exist_ok=True)


def load_legacy_h5_model(model_path):
    # Reconstruct the known training architecture, then load stored weights.
    model = Sequential(name="sequential")
    model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=(100, 1), name="lstm"))
    model.add(Dropout(0.2, name="dropout"))
    model.add(LSTM(60, activation="relu", return_sequences=True, name="lstm_1"))
    model.add(Dropout(0.3, name="dropout_1"))
    model.add(LSTM(80, activation="relu", return_sequences=True, name="lstm_2"))
    model.add(Dropout(0.4, name="dropout_2"))
    model.add(LSTM(120, activation="relu", return_sequences=False, name="lstm_3"))
    model.add(Dropout(0.5, name="dropout_3"))
    model.add(Dense(1, name="dense"))

    model.load_weights(model_path)
    return model


model = load_legacy_h5_model(
    os.path.join(os.path.dirname(__file__), "stock_dl_model.h5")
)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'POWERGRID.NS'  # Default stock if none is entered
        
        # Define the start and end dates for stock data
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.today()
        
        # Download stock data
        df = yf.download(stock, start=start, end=end)
        
        # Descriptive Data
        data_desc = df.describe()
        
        # Exponential Moving Averages
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()
        
        # Data splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
        
        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        
        # Prepare data for prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        # Make predictions
        y_predicted = model.predict(x_test)
        
        # Inverse scaling for predictions and actual values
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Future forecast: predict daily prices from today through May 31, 2026
        forecast_end = dt.datetime(2026, 5, 31)
        future_dates = pd.bdate_range(start=end + dt.timedelta(days=1), end=forecast_end)
        last_100_scaled = input_data[-100:]  # last 100 days of scaled data
        future_input = last_100_scaled.copy().tolist()
        future_preds_scaled = []
        for _ in range(len(future_dates)):
            seq = np.array(future_input[-100:]).reshape(1, 100, 1)
            next_val = model.predict(seq, verbose=0)[0, 0]
            future_preds_scaled.append([next_val])
            future_input.append([next_val])
        future_preds = scaler.inverse_transform(np.array(future_preds_scaled))

        # ── Next-5-day forecast table data ──────────────────────────────────
        next5_dates  = future_dates[:5].tolist()
        next5_prices = future_preds[:5].flatten().tolist()
        last_close   = float(df['Close'].iloc[-1])
        next5_rows   = []
        prev_price   = last_close
        for d, p in zip(next5_dates, next5_prices):
            change = p - prev_price
            pct    = (change / prev_price) * 100
            next5_rows.append({
                'date':   d.strftime('%a, %b %d %Y'),
                'price':  round(p, 2),
                'change': round(change, 2),
                'pct':    round(pct, 2),
            })
            prev_price = p

        # ─── Chart data for interactive JS charts ──────────────────────────
        step = max(1, len(df) // 700)
        chart_dates  = df.index[::step].strftime('%Y-%m-%d').tolist()
        chart_close  = [round(float(v), 2) for v in df['Close'].iloc[::step].values.flatten()]
        chart_ema20  = [round(float(v), 2) for v in ema20.iloc[::step].values.flatten()]
        chart_ema50  = [round(float(v), 2) for v in ema50.iloc[::step].values.flatten()]
        chart_ema100 = [round(float(v), 2) for v in ema100.iloc[::step].values.flatten()]
        chart_ema200 = [round(float(v), 2) for v in ema200.iloc[::step].values.flatten()]

        pred_step = max(1, len(data_testing) // 500)
        chart_test_dates   = data_testing.index[::pred_step].strftime('%Y-%m-%d').tolist()
        chart_y_test       = [round(float(v), 2) for v in y_test.flatten()[::pred_step]]
        chart_y_pred       = [round(float(v), 2) for v in y_predicted.flatten()[::pred_step]]
        chart_future_dates = [d.strftime('%Y-%m-%d') for d in future_dates]
        chart_future_preds = [round(float(v), 2) for v in future_preds.flatten()]

        # Save dataset as CSV
        csv_file_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_file_path)

        # Return the rendered template with chart data
        return render_template(
            'index.html',
            has_results=True,
            next5_rows=next5_rows,
            chart_dates=json.dumps(chart_dates),
            chart_close=json.dumps(chart_close),
            chart_ema20=json.dumps(chart_ema20),
            chart_ema50=json.dumps(chart_ema50),
            chart_ema100=json.dumps(chart_ema100),
            chart_ema200=json.dumps(chart_ema200),
            chart_test_dates=json.dumps(chart_test_dates),
            chart_y_test=json.dumps(chart_y_test),
            chart_y_pred=json.dumps(chart_y_pred),
            chart_future_dates=json.dumps(chart_future_dates),
            chart_future_preds=json.dumps(chart_future_preds),
            data_desc=data_desc.to_html(classes='pred-table', border=0, float_format=lambda x: '{:,.2f}'.format(x)),
            dataset_link=csv_file_path,
            data_start=start.strftime('%b %Y'),
            data_end=end.strftime('%b %Y')
        )

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)


if __name__ == '__main__':
    threading.Timer(1.2, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    app.run(debug=True)
