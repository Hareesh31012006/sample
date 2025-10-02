# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import requests
from datetime import datetime

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# ---------------------
# Configuration / Keys
# ---------------------
# Use Streamlit secrets or environment variables in production:
ALPHA_VANTAGE_API_KEY = st.secrets.get("QE02OEX4QRX0NYAK") or os.getenv("QE02OEX4QRX0NYAK")

# GNews used here does not require an API key. (If you use other news APIs, store their keys similarly.)
# Example: put ALPHA_VANTAGE_API_KEY in Streamlit secrets or set env var before running:
# export ALPHA_VANTAGE_API_KEY="your_key_here"

# ---------------------
# Utility: Sentiment
# ---------------------
@st.cache_data
def textblob_sentiment(text: str) -> float:
    """Return polarity in range [-1.0, 1.0]. 0 = neutral."""
    try:
        if not isinstance(text, str) or text.strip() == "":
            return 0.0
        tb = TextBlob(text)
        return float(tb.sentiment.polarity)
    except Exception:
        return 0.0

# Optional: switch to HuggingFace pipeline (requires installing transformers & torch).
# We'll keep TextBlob as default to avoid heavy dependencies and 'torch' errors.
USE_HF_OPTION = False  # default False; UI lets user toggle

@st.cache_resource
def load_hf_pipeline():
    """If user chooses HF and packages are installed, this tries to load the HF pipeline."""
    try:
        from transformers import pipeline
        import torch  # ensure torch is imported
        pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        return pipe
    except Exception as e:
        st.warning(f"HuggingFace pipeline not available: {e}")
        return None

def hf_sentiment(pipe, text: str) -> float:
    """Return HF sentiment in [-1,1] (positive->+, negative->-)."""
    try:
        out = pipe(text[:512])[0]
        label = out.get("label", "POSITIVE")
        score = float(out.get("score", 0.0))
        return score if label.upper().startswith("POS") else -score
    except Exception:
        return 0.0

def get_sentiment(text: str, use_hf=False, hf_pipe=None) -> float:
    """Unified sentiment accessor. Prefers HF if enabled and available, else TextBlob."""
    if use_hf and hf_pipe is not None:
        return hf_sentiment(hf_pipe, text)
    return textblob_sentiment(text)


# ---------------------
# Utility: News date column inference
# ---------------------
def infer_date_column(df: pd.DataFrame):
    candidates = ['published date', 'published', 'publishedAt', 'date', 'published_at', 'pubDate']
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive search
    for c in df.columns:
        if 'publish' in c.lower() or 'date' in c.lower() or 'time' in c.lower():
            return c
    return None

# ---------------------
# Core: analyze_stock
# ---------------------
def analyze_stock(ticker_symbol: str, use_hf=False):
    """Fetch stock & news, compute sentiment and a simple prediction for the next close."""
    if not ALPHA_VANTAGE_API_KEY:
        st.error("Alpha Vantage API key not set. Add it to Streamlit secrets or set ALPHA_VANTAGE_API_KEY env var.")
        return None, "HOLD (no API key)"

    # 1) Fetch stock data (Alpha Vantage daily)
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        stock_data, meta = ts.get_daily(symbol=ticker_symbol, outputsize='full')
    except Exception as e:
        st.error(f"Error fetching stock data for {ticker_symbol}: {e}")
        return None, "HOLD (data error)"

    if stock_data.empty:
        st.warning(f"No historical data returned for {ticker_symbol}")
        return None, "HOLD (no data)"

    # normalize index -> column 'date'
    stock_data.index = pd.to_datetime(stock_data.index)
    stock_data.index.name = 'date'
    stock_df = stock_data.reset_index().copy()  # has column 'date' and columns '1. open', '2. high', etc.

    # 2) Fetch news with GNews
    try:
        google_news = GNews(language='en', country='US', period='7d')  # last 7 days; you can change
        raw_articles = google_news.get_news(f"{ticker_symbol} stock")
        news_df = pd.DataFrame(raw_articles)
    except Exception as e:
        st.warning(f"Could not fetch news (proceeding without news): {e}")
        news_df = pd.DataFrame()

    # 3) Compute daily sentiment from news (if any)
    if not news_df.empty:
        date_col = infer_date_column(news_df)
        if date_col is None:
            # fallback: try to parse a 'datetime' like field from 'published date' in various formats
            st.warning("Couldn't find a published-date column in news results; skipping news sentiment.")
            daily_sentiment = pd.DataFrame(columns=['date', 'average_sentiment_score'])
        else:
            # ensure datetime conversion
            news_df[date_col] = pd.to_datetime(news_df[date_col], errors='coerce')
            news_df = news_df.dropna(subset=[date_col])
            # pick text field for sentiment:
            text_col = 'title' if 'title' in news_df.columns else news_df.columns[0]
            # choose HF pipeline if requested
            hf_pipe = None
            if use_hf:
                hf_pipe = load_hf_pipeline()
                if hf_pipe is None:
                    st.info("HuggingFace pipeline requested but could not be loaded. Falling back to TextBlob.")
                    use_hf = False
            # compute sentiment for every article (fast: vectorized apply)
            news_df['sentiment_score'] = news_df[text_col].astype(str).apply(lambda t: get_sentiment(t, use_hf=use_hf, hf_pipe=hf_pipe))
            # aggregate by calendar date (market days)
            news_df['published_date_only'] = news_df[date_col].dt.date
            daily_sentiment = news_df.groupby('published_date_only')['sentiment_score'].mean().reset_index()
            daily_sentiment.rename(columns={'published_date_only': 'date', 'sentiment_score': 'average_sentiment_score'}, inplace=True)
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
    else:
        daily_sentiment = pd.DataFrame(columns=['date', 'average_sentiment_score'])

    # 4) Merge sentiment into stock dataframe (left join on date)
    merged = pd.merge(stock_df, daily_sentiment, on='date', how='left')
    merged['average_sentiment_score'] = merged['average_sentiment_score'].fillna(0.0)

    # 5) Feature engineering (simple)
    try:
        merged = merged.sort_values('date')
        merged['MA_50'] = merged['4. close'].rolling(window=50, min_periods=1).mean()
        merged['MA_200'] = merged['4. close'].rolling(window=200, min_periods=1).mean()
        merged['Daily_Return'] = merged['4. close'].pct_change().fillna(0.0)
        # drop rows if not enough history to compute desired features confidently
        merged = merged.dropna(subset=['MA_50', 'MA_200'])
    except Exception as e:
        st.warning(f"Feature engineering warning: {e}")

    if merged.empty or len(merged) < 10:
        st.warning("Not enough data after preprocessing.")
        return merged, "HOLD (insufficient data)"

    # 6) Modeling: train on all historical rows EXCEPT last row, predict the last row (no leakage)
    features = ['4. close', '5. volume', 'MA_50', 'MA_200', 'Daily_Return', 'average_sentiment_score']
    if not all(f in merged.columns for f in features):
        st.error("Required features missing after preprocessing. Please check data.")
        return merged, "HOLD (missing features)"

    X = merged[features]
    y = merged['4. close']

    # train on everything but the last row
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_last = X.iloc[-1:].copy()

    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_last = float(model.predict(X_last)[0])
    except Exception as e:
        st.error(f"Model training/prediction failed: {e}")
        return merged, "HOLD (model error)"

    # attach predicted last (only one point) into dataframe column for display
    merged['Predicted_Close'] = np.nan
    merged.loc[merged.index[-1], 'Predicted_Close'] = pred_last

    # 7) Investment suggestion using price delta + sentiment
    latest_actual = float(merged.iloc[-1]['4. close'])
    latest_sentiment = float(merged.iloc[-1]['average_sentiment_score'])
    price_diff = pred_last - latest_actual
    price_threshold = latest_actual * 0.01  # 1%

    # Thresholds adapted for sentiment range [-1,1]
    pos_sent = 0.2
    neg_sent = -0.2

    if price_diff > price_threshold and latest_sentiment > pos_sent:
        suggestion = "BUY"
    elif price_diff < -price_threshold and latest_sentiment < neg_sent:
        suggestion = "SELL"
    else:
        suggestion = "HOLD"

    return merged, suggestion


# ---------------------
# Streamlit UI
# ---------------------
st.title("Stock Market Predictor Dashboard (fixed sentiment)")

st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker (Alpha Vantage symbol)", "AAPL").upper()
use_hf = st.sidebar.checkbox("Use HuggingFace sentiment (requires transformers & torch installed)", False)

if st.sidebar.button("Analyze"):
    with st.spinner("Fetching & analyzing..."):
        data, suggestion = analyze_stock(ticker, use_hf=use_hf)

    if data is None:
        st.error("No data returned. Check logs above.")
    elif data.empty:
        st.warning("Data after processing is empty. Try other ticker or check your API key.")
    else:
        st.subheader(f"Investment Suggestion: {suggestion}")
        # Plot: actual close price (all), predicted last as scatter
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data['date'], data['4. close'], label='Actual Close')
        # if Predicted_Close exists for the last row, plot it as a marker
        if data['Predicted_Close'].notnull().any():
            last = data.iloc[-1]
            ax.scatter([last['date']], [last['Predicted_Close']], color='red', s=80, label='Predicted Next Close')
        ax.set_title(f"{ticker} - Actual Close and Predicted (next day)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig)

        # Sentiment plot
        st.subheader("Daily Average Sentiment (news)")
        if 'average_sentiment_score' in data.columns:
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.plot(data['date'], data['average_sentiment_score'], label='avg sentiment')
            ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax2.set_title("Daily Average Sentiment")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Sentiment (TextBlob polarity, -1..1)")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.info("No sentiment column available.")

        # Volume
        st.subheader("Volume")
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        ax3.bar(data['date'], data['5. volume'])
        ax3.set_title("Volume")
        st.pyplot(fig3)

        # Show table
        st.subheader("Last 5 rows")
        display_cols = ['date', '4. close', 'Predicted_Close', 'average_sentiment_score', '5. volume']
        st.dataframe(data[display_cols].tail().reset_index(drop=True))
