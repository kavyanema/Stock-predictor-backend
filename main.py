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

def get_currency_info(symbol):
    symbol_upper = symbol.upper()
   
    # Indian stocks (NSE/BSE)
    if symbol_upper.endswith('.NS') or symbol_upper.endswith('.BO'):
        return "INR", "₹", "India (NSE/BSE)"
   
    # UK stocks
    elif symbol_upper.endswith('.L'):
        return "GBP", "£", "United Kingdom (LSE)"
   
    # European stocks
    elif symbol_upper.endswith('.PA'):  # Paris
        return "EUR", "€", "France (Euronext Paris)"
    elif symbol_upper.endswith('.DE'):  # Germany
        return "EUR", "€", "Germany (XETRA)"
    elif symbol_upper.endswith('.AS'):  # Amsterdam
        return "EUR", "€", "Netherlands (Euronext Amsterdam)"
   
    # Japanese stocks
    elif symbol_upper.endswith('.T'):
        return "JPY", "¥", "Japan (TSE)"
   
    # Hong Kong stocks
    elif symbol_upper.endswith('.HK'):
        return "HKD", "HK$", "Hong Kong (HKEX)"
   
    # Canadian stocks
    elif symbol_upper.endswith('.TO') or symbol_upper.endswith('.V'):
        return "CAD", "C$", "Canada (TSX)"
   
    # Australian stocks
    elif symbol_upper.endswith('.AX'):
        return "AUD", "A$", "Australia (ASX)"
   
    # US stocks (default - no suffix or common US exchanges)
    else:
        return "USD", "$", "United States (NASDAQ/NYSE)"

def fetch_prices(symbol):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "1y", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers, timeout=10)
    data = r.json()

    prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    prices = [p for p in prices if p is not None]
   
    # Also get currency from Yahoo Finance API if available
    try:
        api_currency = data["chart"]["result"][0]["meta"].get("currency", None)
    except:
        api_currency = None

    return np.array(prices), api_currency

def create_features(prices, lookback=10):
    X, y = [], []
   
    for i in range(lookback, len(prices)):
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
        prices, api_currency = fetch_prices(symbol)

        if len(prices) < 50:
            return {"error": "Not enough data"}


        currency_code, currency_symbol, exchange_name = get_currency_info(symbol)
       
        if api_currency:
            currency_code = api_currency.upper()

        lookback = 10
        X, y = create_features(prices, lookback)
       
        if len(X) < 50:
            return {"error": "Not enough data after feature creation"}


        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        current_price = prices[-1]
       
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
       
        max_change = current_price * 0.05
        predicted_change = raw_prediction - current_price
       
        if abs(predicted_change) > max_change:
            predicted_change = np.sign(predicted_change) * max_change
       
        predicted_price = round(current_price + predicted_change, 2)
        current_price = round(current_price, 2)
       
        pct_change = ((predicted_price - current_price) / current_price) * 100

        trend = "Bullish" if predicted_price > current_price else "Bearish" if predicted_price < current_price else "Neutral"

        return {
            "symbol": symbol.upper(),
            "exchange": exchange_name,
            "currency": {
                "code": currency_code,
                "symbol": currency_symbol
            },
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_percent": round(pct_change, 2),
            "trend": trend,
            "accuracy": {
                f"MAE_{currency_code}": round(mae, 2),
                f"RMSE_{currency_code}": round(rmse, 2),
                "MAPE_percent": round(mape, 2)
            },
            "model": "Ridge Regression with Feature Engineering",
            "note": "Prediction limited to realistic daily range (±5%)"
        }

    except Exception as e:
        return {"error": str(e)}
