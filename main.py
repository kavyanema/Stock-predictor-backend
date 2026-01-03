from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
from typing import Dict, List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# News API configuration - Get free API key from https://newsapi.org
NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE"  # Replace with your API key

def get_currency_info(symbol):
    """Get currency and exchange information based on stock symbol"""
    symbol_upper = symbol.upper()
    
    if symbol_upper.endswith('.NS') or symbol_upper.endswith('.BO'):
        return "INR", "₹", "India (NSE/BSE)", "India"
    elif symbol_upper.endswith('.L'):
        return "GBP", "£", "United Kingdom (LSE)", "UK"
    elif symbol_upper.endswith('.PA'):
        return "EUR", "€", "France (Euronext Paris)", "France"
    elif symbol_upper.endswith('.DE'):
        return "EUR", "€", "Germany (XETRA)", "Germany"
    elif symbol_upper.endswith('.AS'):
        return "EUR", "€", "Netherlands (Euronext Amsterdam)", "Netherlands"
    elif symbol_upper.endswith('.T'):
        return "JPY", "¥", "Japan (TSE)", "Japan"
    elif symbol_upper.endswith('.HK'):
        return "HKD", "HK$", "Hong Kong (HKEX)", "Hong Kong"
    elif symbol_upper.endswith('.TO') or symbol_upper.endswith('.V'):
        return "CAD", "C$", "Canada (TSX)", "Canada"
    elif symbol_upper.endswith('.AX'):
        return "AUD", "A$", "Australia (ASX)", "Australia"
    else:
        return "USD", "$", "United States (NASDAQ/NYSE)", "USA"

def get_company_name(symbol):
    """Extract company name from symbol"""
    clean_symbol = symbol.upper()
    for suffix in ['.NS', '.BO', '.L', '.PA', '.DE', '.AS', '.T', '.HK', '.TO', '.V', '.AX']:
        clean_symbol = clean_symbol.replace(suffix, '')
    return clean_symbol

def fetch_news(symbol, country):
    """Fetch political and economic news relevant to the stock"""
    if NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        return {"error": "NEWS_API_KEY not configured. Get free key at https://newsapi.org"}
    
    company_name = get_company_name(symbol)
    news_items = []
    
    try:
        url = "https://newsapi.org/v2/everything"
        
        # Fetch company-specific news
        params = {
            "q": f"{company_name} OR {symbol}",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
            "apiKey": NEWS_API_KEY,
            "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for article in data.get("articles", [])[:5]:
                news_items.append({
                    "title": article["title"],
                    "description": article.get("description", ""),
                    "source": article["source"]["name"],
                    "published": article["publishedAt"],
                    "type": "company"
                })
        elif response.status_code == 401:
            return {"error": "Invalid NEWS_API_KEY. Get free key at https://newsapi.org"}
        elif response.status_code == 429:
            return {"error": "NewsAPI rate limit exceeded. Upgrade your plan or wait."}
        
        # Fetch political/economic news for the country
        political_keywords = f"politics OR economy OR trade OR policy OR election OR government {country}"
        params["q"] = political_keywords
        params["pageSize"] = 3
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for article in data.get("articles", [])[:3]:
                news_items.append({
                    "title": article["title"],
                    "description": article.get("description", ""),
                    "source": article["source"]["name"],
                    "published": article["publishedAt"],
                    "type": "political"
                })
                
    except requests.exceptions.Timeout:
        return {"error": "NewsAPI request timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"NewsAPI request failed: {str(e)}"}
    
    if len(news_items) == 0:
        return {"error": "No news articles found for this symbol"}
    
    return news_items

def analyze_sentiment(news_items: List[Dict]) -> Dict:
    """Analyze sentiment of news articles using keyword matching"""
    positive_keywords = [
        'growth', 'profit', 'gain', 'rise', 'increase', 'surge', 'jump', 'rally',
        'beat', 'exceed', 'strong', 'positive', 'optimistic', 'bullish', 'boom',
        'expansion', 'recovery', 'breakthrough', 'innovation', 'success', 'improve'
    ]
    
    negative_keywords = [
        'loss', 'decline', 'fall', 'drop', 'decrease', 'plunge', 'crash', 'weak',
        'negative', 'pessimistic', 'bearish', 'recession', 'crisis', 'concern',
        'risk', 'threat', 'warning', 'miss', 'disappoint', 'cut', 'layoff', 'debt'
    ]
    
    political_negative = [
        'war', 'conflict', 'sanction', 'tension', 'protest', 'instability',
        'uncertainty', 'regulation', 'restriction', 'tariff', 'investigation'
    ]
    
    political_positive = [
        'peace', 'agreement', 'deal', 'treaty', 'cooperation', 'stability',
        'deregulation', 'stimulus', 'support', 'alliance', 'partnership'
    ]
    
    sentiment_score = 0
    positive_signals = []
    negative_signals = []
    
    for news in news_items:
        text = (news.get('title', '') + ' ' + news.get('description', '')).lower()
        
        # Count positive keywords
        for keyword in positive_keywords:
            if keyword in text:
                sentiment_score += 1
                positive_signals.append(f"{keyword.title()} mentioned in: {news['title'][:50]}...")
        
        # Count negative keywords
        for keyword in negative_keywords:
            if keyword in text:
                sentiment_score -= 1
                negative_signals.append(f"{keyword.title()} mentioned in: {news['title'][:50]}...")
        
        # Political sentiment (weighted more heavily)
        if news['type'] == 'political':
            for keyword in political_positive:
                if keyword in text:
                    sentiment_score += 1.5
                    positive_signals.append(f"Political positive: {keyword.title()} - {news['title'][:50]}...")
            
            for keyword in political_negative:
                if keyword in text:
                    sentiment_score -= 1.5
                    negative_signals.append(f"Political negative: {keyword.title()} - {news['title'][:50]}...")
    
    # Normalize sentiment score
    total_signals = len(positive_signals) + len(negative_signals)
    normalized_score = sentiment_score / max(total_signals, 1)
    
    return {
        "score": normalized_score,
        "positive_signals": positive_signals[:5],
        "negative_signals": negative_signals[:5],
        "total_news_items": len(news_items)
    }

def generate_prediction_reasoning(sentiment, trend, pct_change):
    """Generate human-readable reasoning for the prediction"""
    reasons = []
    
    # Technical analysis reason
    if abs(pct_change) > 2:
        reasons.append(f"Strong technical momentum showing {abs(pct_change):.2f}% expected movement")
    elif abs(pct_change) > 1:
        reasons.append(f"Moderate technical signals indicating {abs(pct_change):.2f}% change")
    else:
        reasons.append("Technical indicators suggest stable price action")
    
    # Sentiment-based reasoning
    if sentiment["score"] > 0.5:
        reasons.append("News sentiment is POSITIVE: " + ", ".join(sentiment["positive_signals"][:2]))
    elif sentiment["score"] < -0.5:
        reasons.append("News sentiment is NEGATIVE: " + ", ".join(sentiment["negative_signals"][:2]))
    else:
        reasons.append("News sentiment is NEUTRAL with mixed signals")
    
    # Overall prediction
    if trend == "Bullish":
        prediction = "LIKELY TO RISE"
        if sentiment["score"] > 0:
            confidence = "High confidence"
            reasons.append("Both technical and news sentiment align bullishly")
        else:
            confidence = "Moderate confidence"
            reasons.append("Technical indicators bullish but news sentiment is cautious")
    elif trend == "Bearish":
        prediction = "LIKELY TO DIP"
        if sentiment["score"] < 0:
            confidence = "High confidence"
            reasons.append("Both technical and news sentiment align bearishly")
        else:
            confidence = "Moderate confidence"
            reasons.append("Technical indicators bearish but news sentiment is positive")
    else:
        prediction = "LIKELY TO REMAIN STABLE"
        confidence = "Moderate confidence"
        reasons.append("Mixed signals suggest sideways movement")
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "reasons": reasons
    }

def fetch_prices(symbol):
    """Fetch historical stock prices from Yahoo Finance"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": "1y", "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, params=params, headers=headers, timeout=10)
    data = r.json()

    prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    prices = [p for p in prices if p is not None]
    
    try:
        api_currency = data["chart"]["result"][0]["meta"].get("currency", None)
    except:
        api_currency = None

    return np.array(prices), api_currency

def create_features(prices, lookback=10, sentiment_score=0):
    """Create features including sentiment score"""
    X, y = [], []
    
    for i in range(lookback, len(prices)):
        window = prices[i-lookback:i]
        
        features = [
            window[-1],
            np.mean(window),
            np.std(window),
            (window[-1] - window[0]) / window[0],
            (window[-1] - window[-5]) / window[-5] if i >= 5 else 0,
            sentiment_score,
        ]
        
        X.append(features)
        y.append(prices[i])
    
    return np.array(X), np.array(y)

@app.get("/predict/{symbol}")
def predict_and_evaluate(symbol: str):
    try:
        # Fetch historical prices
        prices, api_currency = fetch_prices(symbol)

        if len(prices) < 50:
            return {"error": "Not enough data"}

        # Get currency and country information
        currency_code, currency_symbol, exchange_name, country = get_currency_info(symbol)
        
        if api_currency:
            currency_code = api_currency.upper()

        # Fetch and analyze news
        news_data = fetch_news(symbol, country)
        
        # Check if news fetch returned an error
        if isinstance(news_data, dict) and "error" in news_data:
            return {"error": news_data["error"]}
        
        sentiment = analyze_sentiment(news_data)
        
        # Create features with sentiment
        lookback = 10
        X, y = create_features(prices, lookback, sentiment["score"])
        
        if len(X) < 50:
            return {"error": "Not enough data after feature creation"}

        # Train-test split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Evaluate on test set
        predictions = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # Make next-day prediction
        current_price = prices[-1]
        
        recent_window = prices[-lookback:]
        next_features = np.array([[
            recent_window[-1],
            np.mean(recent_window),
            np.std(recent_window),
            (recent_window[-1] - recent_window[0]) / recent_window[0],
            (recent_window[-1] - recent_window[-5]) / recent_window[-5],
            sentiment["score"],
        ]])
        
        next_features_scaled = scaler.transform(next_features)
        raw_prediction = model.predict(next_features_scaled)[0]
        
        # Adjust prediction based on sentiment
        sentiment_adjustment = sentiment["score"] * current_price * 0.02
        raw_prediction += sentiment_adjustment
        
        # Limit to realistic daily range
        max_change = current_price * 0.05
        predicted_change = raw_prediction - current_price
        
        if abs(predicted_change) > max_change:
            predicted_change = np.sign(predicted_change) * max_change
        
        predicted_price = round(current_price + predicted_change, 2)
        current_price = round(current_price, 2)
        
        pct_change = ((predicted_price - current_price) / current_price) * 100

        trend = "Bullish" if predicted_price > current_price else "Bearish" if predicted_price < current_price else "Neutral"
        
        # Generate reasoning
        reasoning = generate_prediction_reasoning(sentiment, trend, pct_change)

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
            "prediction_analysis": reasoning,
            "news_sentiment": {
                "score": round(sentiment["score"], 2),
                "interpretation": "Positive" if sentiment["score"] > 0.2 else "Negative" if sentiment["score"] < -0.2 else "Neutral",
                "positive_signals": sentiment["positive_signals"],
                "negative_signals": sentiment["negative_signals"],
                "news_count": sentiment["total_news_items"]
            },
            "recent_news": [
                {
                    "title": news["title"],
                    "source": news["source"],
                    "type": news["type"]
                } for news in news_data[:5]
            ],
            "accuracy": {
                f"MAE_{currency_code}": round(mae, 2),
                f"RMSE_{currency_code}": round(rmse, 2),
                "MAPE_percent": round(mape, 2)
            },
            "model": "Ridge Regression with News Sentiment Analysis",
            "note": "Prediction incorporates technical analysis and global political/economic news sentiment"
        }

    except Exception as e:
        return {"error": str(e)}
