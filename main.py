from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def create_features(prices, lookback=10):
    """Create features from historical prices"""
    X, y = [], []
    
    for i in range(lookback, len(prices)):
        # Use recent price changes (returns) instead of absolute prices
        window = prices[i-lookback:i]
        
        features = [
            window[-1],  # Most recent price
            np.mean(window),  # Moving average
            np.std(window),  # Volatility
            (window[-1] - window[0]) / window[0],  # Period return
            (window[-1] - window[-5]) / window[-5] if i >= 5 else 0,  # Recent momentum
        ]
        
        X.append(features)
        y.append(prices[i])
    
    return np.array(X), np.array(y)

@app.get("/predict/{symbol}")
def predict_and_evaluate(symbol: str):
    try:
        prices = fetch_prices(symbol)

        if len(prices) < 50:
            return {"error": "Not enough data"}

        # -----------------------
        # CREATE FEATURES
        # -----------------------
        lookback = 10
        X, y = create_features(prices, lookback)
        
        if len(X) < 50:
            return {"error": "Not enough data after feature creation"}

        # -----------------------
        # TRAIN / TEST SPLIT
        # -----------------------
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # -----------------------
        # SCALE FEATURES
        # -----------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -----------------------
        # TRAIN MODEL (Ridge for regularization)
        # -----------------------
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        # -----------------------
        # ACCURACY METRICS
        # -----------------------
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # -----------------------
        # NEXT DAY PREDICTION (CONSERVATIVE)
        # -----------------------
        current_price = prices[-1]
        
        # Create features for next day prediction
        recent_window = prices[-lookback:]
        next_features = np.array([[
            recent_window[-1],
            np.mean(recent_window),
            np.std(recent_window),
            (recent_window[-1] - recent_window[0]) / recent_window[0],
            (recent_window[-1] - recent_window[-5]) / recent_window[-5],
        ]])
        
        next_features_scaled = scaler.transform(next_features)
        raw_prediction = model.predict(next_features_scaled)[0]
        
        # APPLY CONSERVATIVE DAMPENING
        # Limit prediction to +/- 5% of current price
        max_change = current_price * 0.05
        predicted_change = raw_prediction - current_price
        
        # Dampen extreme predictions
        if abs(predicted_change) > max_change:
            predicted_change = np.sign(predicted_change) * max_change
        
        predicted_price = round(current_price + predicted_change, 2)
        current_price = round(current_price, 2)
        
        # Calculate percentage change
        pct_change = ((predicted_price - current_price) / current_price) * 100

        trend = "Bullish" if predicted_price > current_price else "Bearish" if predicted_price < current_price else "Neutral"

        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_percent": round(pct_change, 2),
            "trend": trend,
            "accuracy": {
                "MAE_rupees": round(mae, 2),
                "RMSE_rupees": round(rmse, 2),
                "MAPE_percent": round(mape, 2)
            },
            "model": "Ridge Regression with Feature Engineering",
            "note": "Prediction limited to realistic daily range (±5%)"
        }

    except Exception as e:
        return {"error": str(e)}
