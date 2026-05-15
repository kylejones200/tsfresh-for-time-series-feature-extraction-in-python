"""Generated from Jupyter notebook: TSfresh and Polr

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import pandas as pd

# Create a time series DataFrame
date_range = pd.date_range(start="2023-01-01", periods=10, freq="D")
data = pd.DataFrame({"value": range(10)}, index=date_range)
# Shift and rolling mean
data["shifted"] = data["value"].shift(1)
data["rolling_mean"] = data["value"].rolling(window=3).mean()
print(data)
import statsmodels.api as sm

# Simulated data
data = sm.datasets.macrodata.load_pandas().data
inflation = data["infl"]
# Fit ARIMA model
model = sm.tsa.ARIMA(inflation, order=(1, 1, 1))
results = model.fit()
print(results.summary())
# Create example time series data
df = pd.DataFrame(
    {
        "id": [1] * 10 + [2] * 10,
        "time": list(range(10)) + list(range(10)),
        "value": range(20),
    }
)


from prophet import Prophet

# Create example time series data
df = pd.DataFrame(
    {"ds": pd.date_range(start="2023-01-01", periods=100, freq="D"), "y": range(100)}
)
# Fit model
model = Prophet()
model.fit(df)
# Make future predictions
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[-1])
        return x


# Initialize model
model = LSTMModel(input_size=1, hidden_size=10, output_size=1)
print(model)
import polars as pl

# Create a time series DataFrame
df = pl.DataFrame(
    {"timestamp": pl.date_range("2023-01-01", "2023-01-10", "1d"), "value": range(10)}
)
# Compute rolling mean
df = df.with_columns(df["value"].rolling_mean(window_size=3).alias("rolling_mean"))
print(df)


# --- code cell ---

# !pip install tsfresh  # Jupyter-only


# --- code cell ---

import pandas as pd
import polars as pl
import statsmodels.api as sm
import torch.nn as nn
from prophet import Prophet

# Load dataset
file_path = "north_dakota_oil_price.csv"
data = pd.read_csv(file_path, parse_dates=["Date"])
data.set_index("Date", inplace=True)

# Shift and rolling mean
data["shifted"] = data["Oil_Price"].shift(1)
data["rolling_mean"] = data["Oil_Price"].rolling(window=3).mean()
print(data.head())

# ARIMA model
inflation = data["Oil_Price"].dropna()
model = sm.tsa.ARIMA(inflation, order=(1, 1, 1))
results = model.fit()
print(results.summary())


# Prophet model
df_prophet = data.reset_index().rename(columns={"Date": "ds", "Oil_Price": "y"})
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[-1])
        return x


# Initialize model
model = LSTMModel(input_size=1, hidden_size=10, output_size=1)
print(model)

# Polars rolling mean
df_polars = pl.read_csv(file_path)
df_polars = df_polars.with_columns(
    pl.col("Oil_Price").rolling_mean(window_size=3).alias("rolling_mean")
)
print(df_polars.head())
