import requests
import os
from crewai.tools import BaseTool
from utils import get_alphavantage_key, get_newsapi_key, get_twitter_key, get_federalreserve_key
from typing import Optional
# from langchain.tools import BaseTool

# class FinBERTTool(BaseTool):
#     name = "financial_sentiment_analyzer"
#     description = "Analyzes the sentiment of financial text"
#     pipeline = None
    
#     def __init__(self, pipeline):
#         self.pipeline = pipeline
#         super().__init__()
    
#     def _run(self, text: str) -> str:
#         results = self.pipeline(text)
#         # Format results into a readable string
#         sentiment_scores = results[0]
#         formatted_result = "\n".join([f"{score['label']}: {score['score']:.4f}" for score in sentiment_scores])
#         return formatted_result


class YahooMarketAPI(BaseTool):
    name: str = "Yahoo Finance Market Data"
    description: str = (
        "Retrieves real-time and historical stock market data, including OHLCV (Open, High, Low, Close, Volume) "
        "information. Useful for tracking price movements, analyzing trends, and making informed trading decisions."
    )

    def _run(self, query: str):
        api_url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={query}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            return data['quoteResponse']['result']
        return "Failed to retrieve stock data."


class NewsAPI(BaseTool):
    name: str = "Financial News Aggregator"
    description: str = (
        "Fetches the latest financial news articles from multiple sources. "
        "Useful for staying updated on market trends, economic events, and industry insights."
    )

    def _run(self, query: str):
        api_key = get_newsapi_key()  # Replace with your API key
        api_url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
        response = requests.get(api_url)
        if response.status_code == 200:
            articles = response.json()["articles"]
            return [{"title": article["title"], "url": article["url"]} for article in articles]
        return "Failed to retrieve news data."


class AlphaVantageAPI(BaseTool):
    name: str = "Alpha Vantage Market Data"
    description: str = (
        "Provides financial data including real-time and historical stock prices, market indicators, and sector-wise analysis. "
        "Supports technical and fundamental analysis for smarter investment decisions."
    )

    def _run(self, query: str):
        api_key = get_alphavantage_key()
        base_url = "https://www.alphavantage.co/query"
        
        # Parse the query to extract function and symbol
        parts = query.split()
        if len(parts) < 2:
            return "Invalid query. Please specify function and symbol."
        
        function = parts[0]
        symbol = parts[1]
        
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data
        return f"Failed to retrieve data from Alpha Vantage. Status code: {response.status_code}"



class TwitterAPI(BaseTool):
    name: str = "Twitter Data API"
    description: str = (
        "Fetches real-time tweets, user details, and trending topics from Twitter. "
        "Supports querying recent tweets based on keywords, fetching user details, and getting trending topics."
    )

    def _run(self, query: str, search_type: str = "tweets", count: int = 10) -> Optional[dict]:
        """
        Fetches data from Twitter API.
        
        Args:
            query (str): Search keyword or username.
            search_type (str): Type of search - 'tweets' for recent tweets, 'user' for user details, 'trending' for trends.
            count (int): Number of tweets to fetch (applicable for 'tweets' search).

        Returns:
            dict: JSON response from Twitter API.
        """
        api_key = get_twitter_key()  # Function to retrieve API key
        headers = {"Authorization": f"Bearer {api_key}"}
        
        base_url = "https://api.twitter.com/2/"
        
        if search_type == "tweets":
            url = f"{base_url}tweets/search/recent"
            params = {"query": query, "max_results": count}
        
        elif search_type == "user":
            url = f"{base_url}users/by/username/{query}"
            params = {}

        elif search_type == "trending":
            url = f"{base_url}trends/place"
            params = {"id": query}  # Query should be WOEID (Where On Earth IDentifier)

        else:
            return "Invalid search type. Use 'tweets', 'user', or 'trending'."
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        
        return f"Failed to retrieve data from Twitter API. Status code: {response.status_code}"
    

class FederalReserveAPI(BaseTool):
    name: str = "Federal Reserve Economic Data"
    description: str = (
        "Fetches macroeconomic data including interest rates, inflation, GDP, and other economic indicators from the Federal Reserve (FRED) API."
    )

    def _run(self, query: str):
        api_key = get_federalreserve_key()  # Function to retrieve API key
        base_url = "https://api.stlouisfed.org/fred/series/observations"

        # Extract series ID from the query
        parts = query.split()
        if len(parts) < 1:
            return "Invalid query. Please specify a valid FRED series ID."
        
        series_id = parts[0]  # The first part of the query should be the series ID

        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json"
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data
        return f"Failed to retrieve data from the Federal Reserve. Status code: {response.status_code}"
    


