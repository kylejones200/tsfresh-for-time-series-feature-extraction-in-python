"""Generated from Jupyter notebook: TSfresh and Polr"""

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_a_time_series_dataframe() -> None:
    date_range = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = pd.DataFrame({"value": range(10)}, index=date_range)
    data["shifted"] = data["value"].shift(1)
    data["rolling_mean"] = data["value"].rolling(window=3).mean()
    print(data)
    macro = sm.datasets.macrodata.load_pandas().data
    inflation = macro["infl"]
    results = sm.tsa.ARIMA(inflation, order=(1, 1, 1)).fit()
    print(results.summary())


def load_dataset() -> None:
    rng = pd.date_range(start="2023-01-01", periods=100, freq="D")
    prices = pd.Series(range(100), index=rng, dtype=float) + pd.Series(
        np.sin(np.linspace(0, 6, 100)) * 5, index=rng
    )
    data = pd.DataFrame({"Date": rng, "Oil_Price": prices.values})
    data.set_index("Date", inplace=True)
    print(data.head())
    inflation = data["Oil_Price"].dropna()
    print(sm.tsa.ARIMA(inflation, order=(1, 1, 1)).fit().summary())
    df_prophet = data.reset_index().rename(columns={"Date": "ds", "Oil_Price": "y"})
    try:
        from prophet import Prophet

        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
    except ImportError:
        print("Prophet not installed; skipped Prophet forecast")
    print(LSTMModel(input_size=1, hidden_size=10, output_size=1))
    df_polars = pl.DataFrame({"Date": rng, "Oil_Price": prices.values})
    print(
        df_polars.with_columns(
            pl.col("Oil_Price").rolling_mean(window_size=3).alias("rolling_mean")
        ).head()
    )


def main() -> None:
    create_a_time_series_dataframe()
    load_dataset()


if __name__ == "__main__":
    main()
