from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_prices(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "1y", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers, timeout=10)
    data = r.json()

    prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    prices = [p for p in prices if p is not None]

    return np.array(prices)

@app.get("/predict/{symbol}")
def predict_and_evaluate(symbol: str):
    try:
        prices = fetch_prices(symbol)

        if len(prices) < 50:
            return {"error": "Not enough data"}

        # -----------------------
        # TRAIN / TEST SPLIT
        # -----------------------
        split = int(len(prices) * 0.8)
        train_prices = prices[:split]
        test_prices = prices[split:]

        X_train = np.arange(len(train_prices)).reshape(-1, 1)
        y_train = train_prices

        X_test = np.arange(len(train_prices), len(prices)).reshape(-1, 1)
        y_test = test_prices

        # -----------------------
        # TRAIN ML MODEL
        # -----------------------
        model = LinearRegression()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # -----------------------
        # ACCURACY METRICS
        # -----------------------
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # -----------------------
        # NEXT DAY PREDICTION
        # -----------------------
        next_day = np.array([[len(prices)]])
        predicted_price = round(model.predict(next_day)[0], 2)
        current_price = round(prices[-1], 2)

        trend = "Bullish" if predicted_price > current_price else "Bearish"

        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "predicted_price": predicted_price,
            "trend": trend,
            "accuracy": {
                "MAE_rupees": round(mae, 2),
                "RMSE_rupees": round(rmse, 2),
                "MAPE_percent": round(mape, 2)
            },
            "model": "Linear Regression"
        }

    except Exception as e:
        return {"error": str(e)}

