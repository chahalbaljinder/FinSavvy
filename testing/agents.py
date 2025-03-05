# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
from utils import get_serper_key
os.environ["SERPER_API_KEY"] = get_serper_key()

from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from tools import YahooMarketAPI, AlphaVantageAPI, TwitterAPI, NewsAPI, FederalReserveAPI

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
Yahoo_Tool = YahooMarketAPI()
AlphaVantage_Tool = AlphaVantageAPI()
X_Tool = TwitterAPI()
News_Tool = NewsAPI()
Federal_Tool = FederalReserveAPI()

market_data_agent = Agent(
    role="Market Data Agent",
    goal=(
        "Fetch, aggregate, and analyze real-time and historical market data, "
        "including OHLCV (Open, High, Low, Close, Volume), technical indicators, "
        "news sentiment, and cryptocurrency market trends to assist traders and investors."
    ),
    backstory=(
        "This agent is designed to provide comprehensive market intelligence by gathering financial data from "
        "multiple sources. It integrates with Yahoo Finance, Alpha Vantage, TradingView, and Binance to collect "
        "OHLCV data, fundamental indicators, and crypto trends. Additionally, it leverages news analysis and "
        "social media sentiment tracking to identify market-moving events, offering actionable insights for investors."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[
        search_tool,            # General market-related searches
        scrape_tool,            # Web scraping for financial insights
        Yahoo_Tool,             # Stock market and fundamental data
        AlphaVantage_Tool,      # Technical indicators & OHLCV data
        X_Tool,                 # Social sentiment analysis from X (formerly Twitter)
        News_Tool,              # Real-time financial news updates
        Federal_Tool,           # Macroeconomic data and policy updates
    ]
)


sentiment_analysis_agent = Agent(
    role="Sentiment Analysis Agent",
    goal=(
        "Monitor and analyze market sentiment by processing news, social media discussions, "
        "and financial reports to detect trends, assess investment risks, and predict market movements."
    ),
    backstory=(
        "This agent specializes in financial sentiment analysis by aggregating and analyzing data "
        "from various sources, including real-time news, social media, and economic reports. "
        "By leveraging News APIs, X (formerly Twitter), Yahoo Finance, and Bloomberg, it classifies "
        "market sentiment as Bullish, Bearish, or Neutral, offering traders and investors valuable insights "
        "to refine their decision-making."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[
        search_tool,            # Find relevant market sentiment data
        scrape_tool,            # Extract sentiment-related financial insights
        Yahoo_Tool,             # Access financial reports and fundamental data
        AlphaVantage_Tool,      # Retrieve stock market trends and indicators
        X_Tool,                 # Analyze social sentiment from Twitter/X
        News_Tool,              # Process real-time financial news sentiment
        Federal_Tool,           # Incorporate macroeconomic sentiment signals
    ]
)


macro_economic_agent = Agent(
    role="Macro-Economic Agent",
    goal=(
        "Track and analyze macroeconomic indicators, including interest rates, inflation, "
        "GDP growth, and global economic policies, to predict market impact and guide "
        "investment strategies with defensive asset recommendations during downturns."
    ),
    backstory=(
        "This agent specializes in macroeconomic analysis, aggregating and interpreting key policy data, "
        "global economic trends, and financial stability indicators. By leveraging real-time data from "
        "the Federal Reserve API, World Bank API, IMF, and market intelligence sources, it provides "
        "actionable insights to help investors anticipate risks, optimize asset allocation, and refine "
        "portfolio strategies in response to economic fluctuations."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[
        search_tool,            # Fetch macroeconomic reports and insights
        scrape_tool,            # Extract policy updates and financial data
        Yahoo_Tool,             # Retrieve financial market trends
        AlphaVantage_Tool,      # Access economic indicators and forex data
        X_Tool,                 # Analyze macroeconomic sentiment from social media
        News_Tool,              # Process real-time economic news
        Federal_Tool,           # Incorporate Federal Reserve and global economic data
    ]
)


forex_agent = Agent(
    role="Forex Agent",
    goal=(
        "Analyze foreign exchange markets, track currency fluctuations, and provide forex "
        "risk management strategies while identifying optimal international investment opportunities."
    ),
    backstory=(
        "This agent specializes in forex market analysis, monitoring exchange rates and currency trends "
        "from multiple global sources. By integrating real-time data from the OANDA API, XE API, and financial "
        "market feeds, it assesses currency volatility, geopolitical risks, and macroeconomic factors affecting forex. "
        "It assists investors, traders, and businesses in hedging forex risk and making data-driven currency investment decisions."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[
        search_tool,            # Retrieve real-time forex reports and currency trends
        scrape_tool,            # Extract currency data from financial sources
        Yahoo_Tool,             # Access historical and real-time forex data
        AlphaVantage_Tool,      # Get exchange rate analytics and economic indicators
        X_Tool,                 # Analyze forex sentiment from social media
        News_Tool,              # Fetch real-time economic and forex-related news
        Federal_Tool,           # Incorporate central bank policies and macroeconomic updates
    ]
)


technical_analysis_agent = Agent(
    role="Technical Analysis Agent",
    goal=(
        "Perform stock market analysis using technical indicators, fundamental valuation metrics, "
        "and stock screening tools to generate confidence scores for investments."
    ),
    backstory=(
        "This agent specializes in technical stock analysis by computing key indicators such as "
        "RSI, MACD, Moving Averages, and Bollinger Bands. It also performs fundamental valuation using "
        "metrics like P/E ratio, EPS, and Book Value. By leveraging real-time data from TradingView API, "
        "Yahoo Finance API, and Screener.in API, it identifies support/resistance levels, trend patterns, "
        "and overall investment confidence scores. Additionally, it integrates financial news and macroeconomic "
        "indicators to refine market insights and improve investment decisions."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[
        search_tool,            # Retrieve real-time stock market reports
        scrape_tool,            # Extract stock price and company data
        Yahoo_Tool,             # Access historical and real-time stock data
        AlphaVantage_Tool,      # Fetch technical indicators and fundamental metrics
        X_Tool,                 # Analyze market sentiment from social media
        News_Tool,              # Fetch real-time financial news and stock trends
        Federal_Tool,           # Incorporate macroeconomic factors affecting stocks
    ]
)


portfolio_strategy_agent = Agent(
    role="Portfolio Strategy Agent",
    goal=(
        "Develop and optimize trading strategies by analyzing historical data and applying portfolio models "
        "to maximize returns while minimizing risk."
    ),
    backstory=(
        "This agent specializes in portfolio strategy by leveraging historical market data and quantitative models. "
        "It suggests trading strategies such as momentum, mean reversion, and value investing. "
        "Additionally, it optimizes investment portfolios using Modern Portfolio Theory (MPT) to achieve maximum returns "
        "with minimal risk. By integrating data from QuantConnect API and MPT models, it computes expected returns based on user input."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[search_tool,
        scrape_tool,
        Yahoo_Tool,
        AlphaVantage_Tool,
        X_Tool,
        News_Tool,
        Federal_Tool
    ]
)

risk_management_agent = Agent(
    role="Advanced Risk Advisor",
    goal=(
        "Conduct real-time and historical risk analysis using financial models, "
        "news sentiment, market volatility, and economic indicators. "
        "Evaluate risk exposure, predict potential threats, and provide risk-adjusted strategies "
        "to ensure alignment with the firm's risk tolerance and investment objectives."
    ),
    backstory=(
        "Equipped with expertise in risk modeling, quantitative finance, and market analysis, "
        "this agent evaluates financial risks using Monte Carlo simulations, Value at Risk (VaR), "
        "and stress testing techniques. It continuously scans market data, central bank policies, "
        "and news sentiment to assess risk levels. "
        "The agent also monitors economic indicators such as inflation, interest rates, "
        "and GDP growth to provide a holistic risk outlook."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[
        search_tool,            # General web search for risk insights
        scrape_tool,            # Web scraping for economic indicators and financial news
        Yahoo_Tool,             # Stock, crypto, and forex data (volatility, price trends)
        AlphaVantage_Tool,      # Fundamental and technical data for risk evaluation
        X_Tool,                 # Real-time market sentiment analysis from financial Twitter (X)
        News_Tool,              # Latest financial news for macro and micro risk insights
        Federal_Tool            # U.S. Federal Reserve data (interest rates, monetary policy)
    ]
)


market_data_task = Task(
    description=(
        "Gather, analyze, and report real-time and historical market data "
        "for the selected asset ({asset_selection}). "
        "Leverage technical indicators, fundamental data, and sentiment analysis "
        "to provide actionable market insights."
    ),
    expected_output=(
        "A comprehensive report on {asset_selection}, including OHLCV trends, "
        "technical indicators, market sentiment, and potential trading opportunities."
    ),
    agent=market_data_agent,
)

sentiment_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market sentiment for the selected asset ({asset_selection}). "
        "Leverage real-time news, social media discussions, and financial reports to classify sentiment "
        "as Bullish, Bearish, or Neutral."
    ),
    expected_output=(
        "A detailed sentiment analysis report on {asset_selection}, including aggregated news sentiment, "
        "social media sentiment trends, and an assessment of potential market impact."
    ),
    agent=sentiment_analysis_agent,
)

# Task for Macro Economic Analysis
macro_economic_task = Task(
    description=(
        "Monitor and analyze macroeconomic indicators such as interest rates, inflation, GDP growth, and "
        "global economic policies. Assess their potential impact on financial markets and investment strategies."
    ),
    expected_output=(
        "A comprehensive macroeconomic report covering key indicators, policy shifts, and market impact analysis. "
        "Includes defensive asset recommendations for potential downturns."
    ),
    agent=macro_economic_agent,
)

# Task for Forex Analysis
forex_analysis_task = Task(
    description=(
        "Continuously monitor and analyze forex market trends, currency fluctuations, and macroeconomic factors "
        "affecting foreign exchange rates. Assess geopolitical risks and provide forex risk management strategies "
        "for the selected asset ({asset_selection})."
    ),
    expected_output=(
        "A detailed forex market report with currency trend analysis, volatility assessment, "
        "and optimal trading or hedging recommendations."
    ),
    agent=forex_agent,
)

# Task for Technical Analysis
technical_analysis_task = Task(
    description=(
        "Analyze stock market trends for {asset_selection} using technical indicators (RSI, MACD, Moving Averages, "
        "Bollinger Bands) and fundamental valuation metrics (P/E ratio, EPS, Book Value). Identify support/resistance "
        "levels, trend patterns, and generate confidence scores."
    ),
    expected_output=(
        "A comprehensive technical analysis report for {asset_selection} with stock trend predictions, "
        "support/resistance insights, and confidence scores."
    ),
    agent=technical_analysis_agent,
)

# Task for Portfolio Strategy Development
portfolio_strategy_task = Task(
    description=(
        "Analyze historical market data for {asset_selection} and apply portfolio optimization models "
        "to develop trading strategies. Evaluate strategies like momentum trading, mean reversion, and value investing. "
        "Optimize portfolio allocation using Modern Portfolio Theory (MPT) while considering risk tolerance ({risk_tolerance})."
    ),
    expected_output=(
        "A portfolio strategy report detailing optimal asset allocation, expected returns, risk assessment, "
        "and recommended trading strategies."
    ),
    agent=portfolio_strategy_agent,
)

# Task for Risk Management
risk_management_task = Task(
    description=(
        "Conduct real-time and historical risk assessments for {asset_selection} using Monte Carlo simulations, "
        "Value at Risk (VaR), and stress testing techniques. Analyze market volatility, macroeconomic indicators, "
        "and news sentiment (News Impact Consideration: {news_impact_consideration}) to evaluate risk exposure."
    ),
    expected_output=(
        "A comprehensive risk assessment report detailing financial threats, VaR calculations, stress test results, "
        "and risk-adjusted recommendations aligned with investment objectives."
    ),
    agent=risk_management_agent,
)

# from crewai import Crew, Process
# from langchain.llms import HuggingFacePipeline
# from transformers import pipeline
# from langchain_ollama import ChatOllama
# from langchain_community.chat_models import ChatOllama


# # Set your Hugging Face API token (replace with your token)
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACE_API_KEY"

# # # Load FinBERT model from Hugging Face for financial sentiment analysis
# # finbert_pipeline = pipeline("text-classification", model="ProsusAI/finbert")

# # Define the manager LLM using FinBERT
# manager_llm = ChatOllama(model="deepseek-r1:1.5b")

# # Alternative lightweight model (if needed)
# # distilbert_pipeline = pipeline("fill-mask", model="distilbert-base-uncased")
# # manager_llm = HuggingFacePipeline(pipeline=distilbert_pipeline)

# # Define the trading crew with specialized agents
# financial_advising_crew = Crew(
#     agents=[
#         market_data_agent,         # Gathers & analyzes real-time market data
#         sentiment_analysis_agent,  # Conducts sentiment analysis on financial news
#         macro_economic_agent,      # Tracks macroeconomic trends & policies
#         forex_agent,               # Analyzes forex markets & currency trends
#         technical_analysis_agent,  # Performs stock analysis using indicators
#         portfolio_strategy_agent,  # Develops portfolio models for max returns
#         risk_management_agent      # Evaluates financial risks & mitigations
#     ],
#     tasks=[
#         market_data_task,          # Analyzes market trends & financial data
#         sentiment_analysis_task,   # Extracts sentiment insights from news/socials
#         macro_economic_task,       # Tracks economic shifts impacting markets
#         forex_analysis_task,       # Predicts forex movements for strategy planning
#         technical_analysis_task,   # Uses TA indicators for stock forecasting
#         portfolio_strategy_task,   # Builds diversified investment strategies
#         risk_management_task       # Identifies & mitigates trading risks
#     ],
#     manager_llm=manager_llm,       # Uses FinBERT for financial decision-making
#     process=Process.hierarchical,  # Implements structured decision-making
#     verbose=True
# )

# # # Execute the financial advising crew
# # financial_advising_crew.kickoff()


# from litellm import completion

# response = completion(
#     model="ollama/llama2", 
#     messages=[{ "content": "respond in 20 words. who are you?","role": "user"}]
#     )

from crewai import Crew, Process
from langchain_ollama import OllamaLLM
from litellm import completion
from langchain_community.chat_models import ChatOllama


from crewai import Crew, Process
from litellm import completion
import os

# Set environment variables
os.environ['GEMINI_API_KEY'] = ""  # Replace with your actual key

# Create a LiteLLM-compatible agent class
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

# class LiteLLMWrapper(LLM):
#     model_name: str
#     temperature: float = 0.7
    
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         messages = [{"role": "user", "content": prompt}]
#         response = completion(
#             model=self.model_name,
#             messages=messages,
#             temperature=self.temperature,
#             stop=stop
#         )
#         return response.choices[0].message.content
    
#     @property
#     def _llm_type(self) -> str:
#         return "litellm"
    
#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         return {"model_name": self.model_name}


# Create a custom LLM class
class CustomLLM(LLM):
    model_name: str = "ollama/llama2"
    temperature: float = 0.7
    max_tokens: int = 1024
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            messages = [{"role": "user", "content": prompt}]
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error processing with {self.model_name}: {str(e)}"
    
    @property
    def _llm_type(self) -> str:
        return "custom_llm"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature}

# Initialize your LiteLLM model (using Gemini)
# manager_llm = LiteLLMWrapper(model_name="gemini/gemini-pro", temperature=0.7)

manager_llm = CustomLLM

# Define the LLM manager using DeepSeek
#manager_llm = ChatOllama(model="deepseek-r1:1.5b")  # You can use another local model if needed

# Define Crew with manager_llm
financial_advising_crew = Crew(
    agents=[
        market_data_agent,
        sentiment_analysis_agent,
        macro_economic_agent,
        forex_agent,
        technical_analysis_agent,
        portfolio_strategy_agent,
        risk_management_agent
    ],
    tasks=[
        market_data_task,
        sentiment_analysis_task,
        macro_economic_task,
        forex_analysis_task,
        technical_analysis_task,
        portfolio_strategy_task,
        risk_management_task
    ],
    manager_llm=manager_llm,  # Uses DeepSeek model for decision-making
    process=Process.hierarchical,
    verbose=True
)

# # Run Crew
# result = financial_advising_crew.kickoff()
# print(result)


# Example data for kicking off the process
financial_trading_inputs = {
    'asset_selection': 'AAPL',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
}

### this execution will take some time to run
result = financial_advising_crew.kickoff(inputs=financial_trading_inputs)
print(result)

from IPython.display import Markdown
Markdown(result.raw)

