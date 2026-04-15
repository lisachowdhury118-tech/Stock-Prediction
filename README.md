# StockVision AI — Deep Learning Stock Price Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.x-black?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Chart.js-4.x-FF6384?style=for-the-badge&logo=chart.js&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **StockVision AI** is a production-style full-stack web application that uses a multi-layer **LSTM (Long Short-Term Memory)** deep learning neural network to analyze historical stock data, predict past closing prices, and generate rolling **future price forecasts** — all served through a sleek, interactive dark-themed dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Live Features](#2-live-features)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Project Structure](#5-project-structure)
6. [How It Works — End-to-End Pipeline](#6-how-it-works--end-to-end-pipeline)
   - [6.1 Data Ingestion](#61-data-ingestion)
   - [6.2 Feature Engineering & Preprocessing](#62-feature-engineering--preprocessing)
   - [6.3 LSTM Model Architecture](#63-lstm-model-architecture)
   - [6.4 Prediction & Inverse Scaling](#64-prediction--inverse-scaling)
   - [6.5 Future Forecasting (Autoregressive)](#65-future-forecasting-autoregressive)
   - [6.6 Web Layer & Rendering](#66-web-layer--rendering)
7. [Model Details](#7-model-details)
8. [Installation & Setup](#8-installation--setup)
9. [Running the Application](#9-running-the-application)
10. [API Routes](#10-api-routes)
11. [Frontend Architecture](#11-frontend-architecture)
12. [Dataset Export](#12-dataset-export)
13. [Model Conversion Utility](#13-model-conversion-utility)
14. [Supported Stocks](#14-supported-stocks)
15. [Configuration & Customization](#15-configuration--customization)
16. [Known Limitations](#16-known-limitations)

---

## 1. Project Overview

StockVision AI allows any user to enter a stock ticker symbol (e.g., `AAPL`, `TSLA`, `POWERGRID.NS`) and instantly receive:

- A **full historical close-price chart** with four Exponential Moving Averages (EMA 20 / 50 / 100 / 200)
- An overlay **Actual vs. Predicted** price chart produced by the trained LSTM model
- A **future price forecast chart** from today through a configurable horizon
- A **next-5-business-day forecast table** with price, change, and percentage movement
- A **descriptive statistics table** for the entire downloaded dataset
- A **CSV download** of the full historical dataset

---

## 2. Live Features

| Feature | Description |
|---|---|
| **Real-time Data** | Pulls live OHLCV data via `yfinance` directly from Yahoo Finance |
| **LSTM Prediction** | 4-layer stacked LSTM with progressive Dropout regularisation |
| **EMA Indicators** | EMA-20, EMA-50, EMA-100, EMA-200 overlaid on the price chart |
| **Future Forecast** | Autoregressive rolling window forecast up to May 31 2026 |
| **5-Day Forecast Table** | Tabulated next-5-day predictions with directional colour coding |
| **Descriptive Statistics** | Count, mean, std, min/max and quartiles rendered as an HTML table |
| **CSV Export** | One-click download of the entire historical dataset |
| **Zoomable Charts** | Pinch/scroll zoom with reset button, powered by chartjs-plugin-zoom |
| **Responsive UI** | Mobile-first grid layout with dark glassmorphism aesthetic |
| **Auto Browser Open** | Flask starts and automatically launches the app in the default browser |

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                         │
│   ┌────────────────────────────────────────────────────────┐     │
│   │  HTML5 + CSS3 (dark theme)  │  Chart.js 4 (canvas)    │     │
│   │  Jinja2 template rendering  │  chartjs-plugin-zoom     │     │
│   └────────────────────────────────────────────────────────┘     │
│                        ▲                 │                        │
│                   HTTP GET/POST          │ JSON chart data        │
└────────────────────────┼─────────────────┼────────────────────────┘
                         │                 │
┌────────────────────────▼─────────────────▼────────────────────────┐
│                     Flask Web Server (app.py)                      │
│                                                                    │
│  Route /           ─── GET  ──►  Render landing page              │
│  Route /           ─── POST ──►  Run full ML pipeline             │
│  Route /download   ─── GET  ──►  Serve CSV file                   │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │                  ML Pipeline (per request)               │     │
│  │                                                          │     │
│  │  yfinance.download() ──► pandas DataFrame                │     │
│  │       │                                                  │     │
│  │       ▼                                                  │     │
│  │  MinMaxScaler (fit on train split 0–70%)                 │     │
│  │       │                                                  │     │
│  │       ▼                                                  │     │
│  │  Sliding window  (sequence length = 100)                 │     │
│  │       │                                                  │     │
│  │       ▼                                                  │     │
│  │  LSTM Model (loaded from stock_dl_model.h5)              │     │
│  │  ┌──────────────────────────────────────────┐           │     │
│  │  │  LSTM(50)  → Dropout(0.2)                │           │     │
│  │  │  LSTM(60)  → Dropout(0.3)                │           │     │
│  │  │  LSTM(80)  → Dropout(0.4)                │           │     │
│  │  │  LSTM(120) → Dropout(0.5)                │           │     │
│  │  │  Dense(1)                                │           │     │
│  │  └──────────────────────────────────────────┘           │     │
│  │       │                                                  │     │
│  │       ▼                                                  │     │
│  │  inverse_transform  ──►  actual price space              │     │
│  │       │                                                  │     │
│  │       ▼                                                  │     │
│  │  Autoregressive forecast loop  ──►  future_preds         │     │
│  └──────────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────────┘
                              │
                    static/  (CSV files)
```

---

## 4. Technology Stack

### Backend

| Package | Version | Role |
|---|---|---|
| **Python** | 3.9+ | Runtime |
| **Flask** | 3.x | Micro web framework, routing, template engine |
| **TensorFlow / Keras** | 2.x | Deep learning model loading & inference |
| **NumPy** | 1.x / 2.x | Numerical array manipulation |
| **pandas** | 2.x | DataFrame operations, CSV I/O |
| **yfinance** | 0.2.x | Yahoo Finance historical data API wrapper |
| **scikit-learn** | 1.x | `MinMaxScaler` for feature normalisation |

### Frontend

| Library | Version | Role |
|---|---|---|
| **Chart.js** | 4.4.7 | Canvas-based interactive line charts |
| **chartjs-adapter-date-fns** | 3.0.0 | Time-scale date parsing for Chart.js |
| **chartjs-plugin-zoom** | 2.0.1 | Pinch/scroll zoom & pan on charts |
| **Hammer.js** | 2.0.8 | Touch gesture recognition (dependency of zoom plugin) |
| **Font Awesome** | 6.4.0 | Icon set |
| **Google Fonts** | — | Inter & Poppins typefaces |
| **Jinja2** | (built into Flask) | Server-side HTML templating |

### Model Training (Jupyter Notebook)

| Tool | Role |
|---|---|
| **Jupyter Notebook** | Interactive training environment |
| **TensorFlow / Keras** | LSTM model definition, compilation, training |
| **scikit-learn** | MinMaxScaler, train/test split logic |
| **matplotlib** | Training visualisations |

---

## 5. Project Structure

```
stock_price_prediction/
│
├── app.py                        # Flask application — main entry point
├── convert_model.py              # Utility: convert legacy .h5 → .keras format
├── stock_dl_model.h5             # Pre-trained LSTM weights (legacy HDF5)
├── stock_dl_model.keras          # Pre-trained LSTM weights (native Keras format)
├── Stock Price Prediction .ipynb # Jupyter notebook — model training & exploration
├── powergrid.csv                 # Raw POWERGRID.NS training data reference
├── README.md                     # This file
│
├── static/                       # Auto-generated CSV dataset downloads
│   ├── AAPL_dataset.csv
│   ├── POWERGRID.NS_dataset.csv
│   └── <TICKER>_dataset.csv      # Created on-the-fly for every query
│
└── templates/
    └── index.html                # Single-page Jinja2 + Chart.js dashboard
```

---

## 6. How It Works — End-to-End Pipeline

### 6.1 Data Ingestion

When a user submits a ticker via the web form, the backend calls:

```python
df = yf.download(stock, start='2000-01-01', end=datetime.today())
```

This fetches daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance, going as far back as January 2000. Only the **Close** price column is used for prediction.

---

### 6.2 Feature Engineering & Preprocessing

```
Full Close series
        │
        ├─── Training set  (0% – 70%)
        │         └── MinMaxScaler.fit_transform()  → [0, 1]
        │
        └─── Testing set   (70% – 100%)
                  └── MinMaxScaler.transform()      → [0, 1]
                        (using the same scaler fitted on train data)
```

The scaler is fitted **only on the training portion** to prevent data leakage. Feature range is normalised to `[0, 1]`.

**Sliding-window sequence construction:**

```
Sequence length = 100 days

x_test[i] = scaled_prices[i-100 : i]    # input window
y_test[i] = scaled_prices[i]            # target (next day)
```

Each input sample is a 100-timestep univariate time series, shaped `(100, 1)`.

---

### 6.3 LSTM Model Architecture

The model is loaded from `stock_dl_model.h5` using a weight-compatible architecture reconstruction:

```
Input shape: (100, 1)
─────────────────────────────────────────────────────────────
Layer               Units   Activation   Dropout   Return Seq
─────────────────────────────────────────────────────────────
LSTM_1              50      ReLU         20%       True
LSTM_2              60      ReLU         30%       True
LSTM_3              80      ReLU         40%       True
LSTM_4              120     ReLU         50%       False
Dense               1       Linear        —        —
─────────────────────────────────────────────────────────────
Output shape: (1,)  — single scalar (normalised next-day close)
```

Progressive Dropout rates (0.2 → 0.5) reduce overfitting, while increasing LSTM units allow the network to capture patterns at multiple levels of abstraction.

---

### 6.4 Prediction & Inverse Scaling

```python
y_predicted = model.predict(x_test)                          # shape (N, 1)
y_predicted = scaler.inverse_transform(y_predicted)          # → price space
y_test      = scaler.inverse_transform(y_test.reshape(-1,1)) # → price space
```

Both predicted and actual arrays are unscaled back to USD/INR prices before being passed to the frontend.

---

### 6.5 Future Forecasting (Autoregressive)

Using the **autoregressive (recursive)** strategy, the model generates future values one step at a time by feeding its own output back as the next input:

```
last_100_days  →  seq[0:100]
      │
      ▼
  model.predict()  →  next_val
      │
      ▼
  append next_val to seq, drop oldest
      │
      ▼
  repeat until forecast horizon reached
```

This produces a rolling forecast from the next business day through **May 31, 2026**, covering every valid trading day (`pd.bdate_range`).

> ⚠️ **Note:** Autoregressive prediction accumulates error over long horizons. The forecast becomes increasingly speculative beyond ~20 trading days and should be treated as a directional signal only.

---

### 6.6 Web Layer & Rendering

All chart data is serialised to JSON and injected into the Jinja2 template:

```python
return render_template(
    'index.html',
    chart_dates        = json.dumps(chart_dates),
    chart_close        = json.dumps(chart_close),
    chart_ema20        = json.dumps(chart_ema20),
    ...
    chart_future_dates = json.dumps(chart_future_dates),
    chart_future_preds = json.dumps(chart_future_preds),
    next5_rows         = next5_rows,          # Python list → Jinja2 loop
    data_desc          = data_desc.to_html(), # pandas → HTML table
)
```

The template reads these JSON blobs directly into Chart.js dataset arrays with no additional AJAX calls after the initial page load.

---

## 7. Model Details

| Property | Value |
|---|---|
| Model type | Stacked LSTM (Recurrent Neural Network) |
| Input window | 100 trading days |
| Input features | Univariate (Close price only) |
| Output | Single-step next-day close (normalised) |
| Training data start | 2000-01-01 |
| Train / Test split | 70% / 30% |
| Normalisation | MinMaxScaler [0, 1] |
| Saved format | HDF5 (`.h5`) + native Keras (`.keras`) |
| Loss function | Mean Squared Error (MSE) |
| Optimizer | Adam |

---

## 8. Installation & Setup

### Prerequisites

- Python **3.9** or higher
- `pip` package manager
- Internet connection (to download live stock data at runtime)

### Clone the Repository

```bash
git clone https://github.com/lisachowdhury118-tech/stock_price_prediction.git
cd stock_price_prediction
```

### Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install flask tensorflow numpy pandas yfinance scikit-learn
```

**Full dependency reference:**

```
flask>=3.0.0
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.31
scikit-learn>=1.3.0
```

---

## 9. Running the Application

```bash
python app.py
```

The application will:

1. Load the pre-trained LSTM weights from `stock_dl_model.h5`
2. Start the Flask development server on `http://127.0.0.1:5000`
3. **Automatically open your default web browser** after a 1.2-second delay

```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

### Using the Dashboard

1. Enter a valid Yahoo Finance ticker in the search box (e.g., `AAPL`, `TSLA`, `MSFT`, `POWERGRID.NS`)
2. Click **Run Analysis**
3. Wait for the loading overlay — the model is fetching data and running inference
4. Explore the interactive charts, scroll/pinch to zoom, view the forecast table and download the dataset

---

## 10. API Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Renders the landing page (no results state) |
| `POST` | `/` | Accepts `stock` form field, runs the full ML pipeline, returns rendered results |
| `GET` | `/download/<filename>` | Streams the requested CSV file from `static/` as a file attachment |

**POST form body:**

```
stock=AAPL
```

If the `stock` field is empty or missing, the application defaults to `POWERGRID.NS`.

---

## 11. Frontend Architecture

The entire UI is a **single Jinja2 template** (`templates/index.html`) containing:

- **Embedded CSS** — ~600 lines of custom CSS with CSS variables-driven dark theme (no external framework)
- **Five Chart.js chart instances** initialised in inline `<script>` blocks:

  | Chart ID | Content |
  |---|---|
  | `closePriceChart` | Full historical close price + EMA-20 / 50 / 100 / 200 |
  | `ema100Chart` | EMA-100 isolated view |
  | `ema200Chart` | EMA-200 isolated view |
  | `predChart` | Actual vs. Predicted prices (test window) |
  | `forecastChart` | Future autoregressive price forecast |

- **Loading overlay** — displayed on form submit, hidden on page reload via CSS class toggle
- **Responsive grid** — 2-column chart layout that collapses to single column below 820 px
- **Directional badge colouring** — green for positive forecast change, red for negative

### Chart Data Flow

```
Flask (Python)            │          Browser (JavaScript)
─────────────────────────────────────────────────────────
json.dumps(chart_dates)  ──►  var chartDates = {{ chart_dates | safe }};
json.dumps(chart_close)  ──►  var chartClose = {{ chart_close | safe }};
                               │
                               ▼
                          new Chart(ctx, { data: { datasets: [{ data: chartClose }] } })
```

---

## 12. Dataset Export

After each analysis, the full historical OHLCV dataset is saved to:

```
static/<TICKER>_dataset.csv
```

A download button on the results page calls `/download/<TICKER>_dataset.csv`, which Flask serves with `Content-Disposition: attachment`.

Pre-generated datasets bundled with the repository:

| File | Ticker |
|---|---|
| `static/AAPL_dataset.csv` | Apple Inc. |
| `static/APLE_dataset.csv` | Apple Hospitality REIT |
| `static/TSLA_dataset.csv` | Tesla Inc. |
| `static/POWERGRID.NS_dataset.csv` | POWERGRID Corporation of India |

---

## 13. Model Conversion Utility

`convert_model.py` is a one-time utility that migrates the legacy HDF5 model to Keras's native format:

```python
from keras.models import load_model

model = load_model("stock_dl_model.h5", compile=False, safe_mode=False)
model.save("stock_dl_model.keras")
```

Run this only if you need to regenerate `stock_dl_model.keras`. The production app (`app.py`) loads the `.h5` weights directly through explicit architecture reconstruction to maintain compatibility across TensorFlow versions.

---

## 14. Supported Stocks

Any ticker symbol supported by Yahoo Finance can be queried, including:

| Exchange | Example Tickers |
|---|---|
| **NASDAQ / NYSE** | `AAPL`, `TSLA`, `MSFT`, `GOOGL`, `AMZN`, `NVDA` |
| **NSE (India)** | `POWERGRID.NS`, `TCS.NS`, `RELIANCE.NS`, `INFY.NS` |
| **BSE (India)** | `POWERGRID.BO`, `TCS.BO` |
| **LSE** | `BP.L`, `SHEL.L` |
| **Indices** | `^GSPC` (S&P 500), `^NSEI` (Nifty 50) |

---

## 15. Configuration & Customization

| Parameter | File | Default | How to Change |
|---|---|---|---|
| Data start date | `app.py` ~line 55 | `2000-01-01` | `dt.datetime(2000, 1, 1)` |
| Forecast end date | `app.py` ~line 89 | `2026-05-31` | `dt.datetime(2026, 5, 31)` |
| LSTM sequence length | `app.py` ~line 70 | `100` | Must match the trained model's input shape |
| Train / test split | `app.py` ~line 64 | `0.70` | Change the `0.70` fraction |
| Chart point limit | `app.py` ~line 104 | `700` | `step = max(1, len(df) // 700)` |
| Flask debug mode | `app.py` ~line 168 | `True` | Set to `False` for production |
| Browser open delay | `app.py` ~line 167 | `1.2 s` | `threading.Timer(1.2, ...)` |

---

## 16. Known Limitations

- **Single feature input** — The model only uses the Close price. Incorporating volume, sentiment analysis, or macroeconomic indicators could significantly improve accuracy.
- **No online retraining** — The model weights are static. The Jupyter notebook must be run manually to retrain with updated data.
- **Autoregressive drift** — Long-horizon forecasts (> 20 trading days) compound prediction errors and should be treated as directional trend signals rather than precise price targets.
- **Yahoo Finance rate limits** — Heavy automated usage may trigger throttling from the `yfinance` data source.
- **Development server only** — The Flask `debug=True` server is not suitable for production. Use Gunicorn or uWSGI behind a reverse proxy for public deployments.

---

## License

This project is released under the **MIT License**.

---

<p align="center">
  Built with TensorFlow · Flask · Chart.js &nbsp;·&nbsp; by <a href="https://github.com/lisachowdhury118-tech">lisachowdhury118-tech</a>
</p>