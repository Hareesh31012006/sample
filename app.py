import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from sklearn.linear_model import LinearRegression
from textblob import TextBlob

st.set_page_config(page_title="Stock Market Predictor", layout="wide")

# -----------------------------
# API Key
# -----------------------------
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")

if not ALPHA_VANTAGE_API_KEY:
    st.error("Alpha Vantage API key not set. Add it to Streamlit secrets or set ALPHA_VANTAGE_API_KEY env var.")
    st.stop()

# -----------------------------
# Sentiment Analysis
# -----------------------------
@st.cache_data
def get_sentiment(text: str) -> float:
    """Returns sentiment score scaled 0-1 using TextBlob."""
    if not isinstance(text, str) or text.strip() == "":
        return 0.5
    tb = TextBlob(text)
    polarity = tb.sentiment.polarity  # -1 to 1
    return (polarity + 1) / 2  # scale 0-1

# -----------------------------
# Stock + News Analysis
# -----------------------------
def analyze_stock(ticker_symbol: str):
    try:
        # Stock data
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        stock_data, _ = ts.get_daily(symbol=ticker_symbol, outputsize='full')
        if stock_data.empty:
            st.warning(f"No stock data for {ticker_symbol}")
            return None, "HOLD"
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data.index.name = "date"
        stock_df = stock_data.reset_index()

        # News
        try:
            gn = GNews(language='en', country='US', period='7d')
            articles = gn.get_news(f"{ticker_symbol} stock")
            news_df = pd.DataFrame(articles)
            if not news_df.empty:
                # determine date column
                date_col = next((c for c in news_df.columns if "publish" in c.lower() or "date" in c.lower()), None)
                if date_col:
                    news_df[date_col] = pd.to_datetime(news_df[date_col], errors='coerce')
                    news_df = news_df.dropna(subset=[date_col])
                    text_col = "title" if "title" in news_df.columns else news_df.columns[0]
                    news_df["sentiment_score"] = news_df[text_col].astype(str).apply(get_sentiment)
                    news_df["date_only"] = news_df[date_col].dt.date
                    daily_sentiment = news_df.groupby("date_only")["sentiment_score"].mean().reset_index()
                    daily_sentiment.rename(columns={"date_only": "date", "sentiment_score": "average_sentiment_score"}, inplace=True)
                    daily_sentiment["date"] = pd.to_datetime(daily_sentiment["date"])
                else:
                    daily_sentiment = pd.DataFrame(columns=["date", "average_sentiment_score"])
            else:
                daily_sentiment = pd.DataFrame(columns=["date", "average_sentiment_score"])
        except Exception:
            daily_sentiment = pd.DataFrame(columns=["date", "average_sentiment_score"])

        # Merge
        merged = pd.merge(stock_df, daily_sentiment, on="date", how="left")
        merged["average_sentiment_score"] = merged["average_sentiment_score"].fillna(0.5)

        # Features
        merged = merged.sort_values("date")
        merged["MA_50"] = merged["4. close"].rolling(50, min_periods=1).mean()
        merged["MA_200"] = merged["4. close"].rolling(200, min_periods=1).mean()
        merged["Daily_Return"] = merged["4. close"].pct_change().fillna(0)

        # Model
        features = ["4. close","5. volume","MA_50","MA_200","Daily_Return","average_sentiment_score"]
        X = merged[features]
        y = merged["4. close"]
        if len(merged) < 2:
            return merged, "HOLD"

        model = LinearRegression()
        model.fit(X.iloc[:-1], y.iloc[:-1])
        pred_last = float(model.predict(X.iloc[-1:].values)[0])
        merged["Predicted_Close"] = np.nan
        merged.loc[merged.index[-1], "Predicted_Close"] = pred_last

        # Suggestion (realistic)
        last_row = merged.iloc[-1]
        diff_pct = (last_row["Predicted_Close"] - last_row["4. close"]) / last_row["4. close"]
        sentiment = last_row["average_sentiment_score"]

        if diff_pct > 0.005 or sentiment > 0.55:
            suggestion = "BUY"
        elif diff_pct < -0.005 or sentiment < 0.45:
            suggestion = "SELL"
        else:
            suggestion = "HOLD"

        return merged, suggestion

    except Exception as e:
        st.error(f"Error analyzing {ticker_symbol}: {e}")
        return None, "HOLD"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Stock Market Predictor Dashboard")

st.sidebar.header("Input")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()

if st.sidebar.button("Analyze"):
    data, suggestion = analyze_stock(ticker)
    if data is None or data.empty:
        st.warning("No data returned. Check API key or ticker symbol.")
    else:
        st.subheader(f"Investment Suggestion: {suggestion}")

        # Plot Close & Prediction
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(data["date"], data["4. close"], label="Actual Close")
        if data["Predicted_Close"].notnull().any():
            last = data.iloc[-1]
            ax.scatter([last["date"]],[last["Predicted_Close"]], color='red', s=80, label="Predicted Next Close")
        ax.set_xlabel("Date"); ax.set_ylabel("Close Price"); ax.legend()
        st.pyplot(fig)

        # Sentiment
        st.subheader("Daily Sentiment")
        if "average_sentiment_score" in data.columns:
            fig2, ax2 = plt.subplots(figsize=(12,4))
            ax2.plot(data["date"], data["average_sentiment_score"], label="avg sentiment")
            ax2.axhline(0.5,color="gray", linestyle="--")
            ax2.set_xlabel("Date"); ax2.set_ylabel("Sentiment (0-1)")
            ax2.legend()
            st.pyplot(fig2)

        # Volume
        st.subheader("Volume")
        fig3, ax3 = plt.subplots(figsize=(12,4))
        ax3.bar(data["date"], data["5. volume"])
        ax3.set_xlabel("Date"); ax3.set_ylabel("Volume")
        st.pyplot(fig3)

        # Table
        st.subheader("Last 5 rows")
        st.dataframe(data[["date","4. close","Predicted_Close","average_sentiment_score","5. volume"]].tail())
