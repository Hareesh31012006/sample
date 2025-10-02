import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import requests # Import requests for handling potential connection errors

# Store API keys (ensure these are handled securely in a real app)
alpha_vantage_api_key = "QE02OEX4QRX0NYAK"
gnews_api_key = "e7f7433291b4e30e20240b7e1781a524"

# Initialize sentiment analyzer
@st.cache_resource
def load_sentiment_analyzer():
    try:
        return pipeline('sentiment-analysis')
    except Exception as e:
        st.error(f"Error loading sentiment analysis model: {e}")
        return None

sentiment_analyzer = load_sentiment_analyzer()

def get_sentiment(text):
    if not isinstance(text, str):
        return None
    max_len = 500 # Approximation of token limit
    if len(text) > max_len:
        text = text[:max_len]
    if sentiment_analyzer is None:
        return None
    try:
        result = sentiment_analyzer(text)
        return result[0]
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return None

def generate_investment_suggestion(actual_close, predicted_close, sentiment_score):
    """Generates an investment suggestion based on price prediction and sentiment."""
    price_difference = predicted_close - actual_close
    price_threshold = actual_close * 0.01 # Define a 1% threshold for "significant" price difference

    # Define sentiment thresholds (adjust as needed)
    positive_sentiment_threshold = 0.6
    negative_sentiment_threshold = 0.4

    if price_difference > price_threshold and sentiment_score > positive_sentiment_threshold:
        return "BUY"
    elif abs(price_difference) <= price_threshold and (sentiment_score >= negative_sentiment_threshold and sentiment_score <= positive_sentiment_threshold):
        return "HOLD"
    elif price_difference < -price_threshold and sentiment_score < negative_sentiment_threshold:
        return "SELL"
    else:
        return "HOLD" # Default to HOLD for other cases


# Function to analyze stock data and news
def analyze_stock(ticker_symbol):
    if not alpha_vantage_api_key or not gnews_api_key:
        st.error("API keys are not set. Please configure them.")
        return None

    try:
        # Data Collection (Stock Data)
        ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
        try:
            stock_data, meta_data = ts.get_daily(symbol=ticker_symbol, outputsize='full')
            if stock_data.empty:
                st.warning(f"No historical stock data found for ticker: {ticker_symbol}")
                return None
            stock_data.index = pd.to_datetime(stock_data.index)
            stock_data.index.name = 'date'
        except Exception as e:
            st.error(f"Error fetching stock data for {ticker_symbol}: {e}")
            return None

        # Data Collection (News Articles)
        try:
            google_news = GNews(language='en', country='US', period='7d') # Fetch news for the last 7 days
            news_articles = google_news.get_news(f'{ticker_symbol} stock')
            news_df = pd.DataFrame(news_articles)

            if news_df.empty:
                st.warning("No recent news found for this ticker. Proceeding with stock data only.")
                merged_data = stock_data.copy()
                merged_data['average_sentiment_score'] = 0 # Assign a neutral sentiment score
            else:
                # Sentiment Analysis
                news_df['sentiment'] = news_df['title'].apply(get_sentiment)
                # Filter out rows where sentiment analysis failed
                news_df_cleaned = news_df.dropna(subset=['sentiment'])

                if not news_df_cleaned.empty:
                    news_df_cleaned['sentiment_label'] = news_df_cleaned['sentiment'].apply(lambda x: x['label'] if x else None)
                    news_df_cleaned['sentiment_score'] = news_df_cleaned['sentiment'].apply(lambda x: x['score'] if x else None)

                    # Convert 'published date' to datetime objects
                    news_df_cleaned['published date'] = pd.to_datetime(news_df_cleaned['published date'])

                    # Group by date and calculate the average sentiment score
                    daily_sentiment = news_df_cleaned.groupby(news_df_cleaned['published date'].dt.date)['sentiment_score'].mean().reset_index()
                    daily_sentiment['published date'] = pd.to_datetime(daily_sentiment['published date'])
                    daily_sentiment.rename(columns={'published date': 'date', 'sentiment_score': 'average_sentiment_score'}, inplace=True)

                    # Merge the average daily sentiment scores with the stock_data DataFrame
                    merged_data = pd.merge(stock_data, daily_sentiment, on='date', how='left')

                    # Handle missing values from the merge using forward fill
                    merged_data['average_sentiment_score'] = merged_data['average_sentiment_score'].ffill()
                    # Fill leading NaNs with a neutral score if ffill couldn't fill everything
                    merged_data['average_sentiment_score'].fillna(0, inplace=True)
                else:
                    st.warning("Sentiment analysis failed for all news articles. Proceeding with stock data only.")
                    merged_data = stock_data.copy()
                    merged_data['average_sentiment_score'] = 0 # Assign a neutral sentiment score

        except requests.exceptions.RequestException as e:
            st.error(f"Network error fetching news for {ticker_symbol}: {e}")
            # Proceed with stock data only in case of network error
            merged_data = stock_data.copy()
            merged_data['average_sentiment_score'] = 0 # Assign a neutral sentiment score
        except Exception as e:
            st.error(f"Error fetching or processing news for {ticker_symbol}: {e}")
            # Proceed with stock data only in case of other news errors
            merged_data = stock_data.copy()
            merged_data['average_sentiment_score'] = 0 # Assign a neutral sentiment score


        # Feature Engineering
        try:
            merged_data['MA_50'] = merged_data['4. close'].rolling(window=50).mean()
            merged_data['MA_200'] = merged_data['4. close'].rolling(window=200).mean()
            merged_data['Daily_Return'] = merged_data['4. close'].pct_change()

            # Handle missing values created by rolling window calculations and the initial merge
            merged_data.dropna(inplace=True)

            if merged_data.empty:
                st.error("Not enough data to generate features after calculating moving averages. Please try a different ticker or date range.")
                return None, None # Return None for both data and suggestion
        except Exception as e:
            st.error(f"Error during feature engineering for {ticker_symbol}: {e}")
            return None, None # Return None for both data and suggestion


        # Model Training and Prediction (Simplified for demonstration - retrain on available data)
        # In a production setting, you would load a pre-trained model.
        try:
            features = ['4. close', '5. volume', 'MA_50', 'MA_200', 'Daily_Return', 'average_sentiment_score']
            target = '4. close' # Predicting the closing price

            # Ensure all required features exist in the DataFrame
            if not all(feature in merged_data.columns for feature in features):
                missing_features = [feature for feature in features if feature not in merged_data.columns]
                st.error(f"Missing required features for model training: {missing_features}")
                return None, None # Return None for both data and suggestion


            X = merged_data[features]
            y = merged_data[target]

            # Train the model on the available data (simplified)
            model = LinearRegression()
            model.fit(X, y)

            # Make predictions on the entire dataset X
            predictions = model.predict(X)

            # Store these predictions in a new column
            merged_data['Predicted_Close'] = predictions

            # Generate investment suggestion
            if not merged_data.empty:
                latest_data = merged_data.iloc[-1]
                latest_actual_close = latest_data['4. close']
                latest_predicted_close = latest_data['Predicted_Close']
                latest_sentiment_score = latest_data['average_sentiment_score']
                investment_suggestion = generate_investment_suggestion(
                    latest_actual_close,
                    latest_predicted_close,
                    latest_sentiment_score
                )
            else:
                 investment_suggestion = "HOLD (Insufficient data)"


        except Exception as e:
            st.error(f"Error during model training or prediction for {ticker_symbol}: {e}")
            return None, None # Return None for both data and suggestion


        return merged_data, investment_suggestion

    except Exception as e:
        st.error(f"An unexpected error occurred during data analysis for {ticker_symbol}: {e}")
        return None, None # Return None for both data and suggestion


st.title('Stock Market Predictor Dashboard')

st.markdown("""
This dashboard visualizes historical stock data, sentiment scores, and predictions.
Enter a stock ticker symbol in the sidebar to view the data.
""")

# Sidebar for user input
st.sidebar.header('User Input')
ticker_symbol = st.sidebar.text_input('Enter Stock Ticker Symbol (e.g., AAPL)', 'AAPL').upper() # Convert to uppercase

# Perform analysis and get data when ticker symbol is entered
if ticker_symbol:
    stock_analysis_data, investment_suggestion = analyze_stock(ticker_symbol)

    if stock_analysis_data is not None and not stock_analysis_data.empty:
        st.subheader(f'Analysis for {ticker_symbol}')

        # Display Investment Suggestion
        st.subheader(f'Investment Suggestion: {investment_suggestion}')

        # Plot Historical Close Price and Predictions
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(stock_analysis_data.index, stock_analysis_data['4. close'], label='Actual Close Price')
        ax1.plot(stock_analysis_data.index, stock_analysis_data['Predicted_Close'], label='Predicted Close Price', linestyle='--')
        ax1.set_title(f'{ticker_symbol} Stock Price: Actual vs Predicted')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price')
        ax1.legend()
        st.pyplot(fig1)

        st.subheader('Daily Sentiment Scores')

        # Plot Daily Sentiment Scores
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(stock_analysis_data.index, stock_analysis_data['average_sentiment_score'], label='Average Sentiment Score', color='orange')
        ax2.set_title(f'{ticker_symbol} Daily Average Sentiment Score')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sentiment Score')
        ax2.legend()
        st.pyplot(fig2)

        st.subheader('Volume')

        # Plot Volume
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.bar(stock_analysis_data.index, stock_analysis_data['5. volume'], label='Volume', color='green')
        ax3.set_title(f'{ticker_symbol} Trading Volume')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volume')
        ax3.legend()
        st.pyplot(fig3)


        st.subheader('Data Table (Last 5 rows)')
        st.dataframe(stock_analysis_data[['4. close', 'Predicted_Close', 'average_sentiment_score', '5. volume']].tail())

    elif stock_analysis_data is not None and stock_analysis_data.empty:
         st.warning(f"The analysis for ticker '{ticker_symbol}' resulted in an empty dataset after processing. This could be due to insufficient data for feature engineering. Please try a different ticker or date range.")
    # Error handling for None return from analyze_stock is implicitly handled by the 'if stock_analysis_data is not None' check.


st.sidebar.markdown("Dashboard created as part of a stock market predictor project.")
