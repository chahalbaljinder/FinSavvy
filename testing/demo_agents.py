# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process
import os
from utils import get_serper_key

# Set environment variables once
os.environ["SERPER_API_KEY"] = get_serper_key()

# Import tools only once
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from tools import YahooMarketAPI, AlphaVantageAPI, TwitterAPI, NewsAPI, FederalReserveAPI
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from litellm import completion

# Initialize all tools at once and store in a dictionary for reuse
def initialize_tools():
    tools = {
        'search': SerperDevTool(),
        'scrape': ScrapeWebsiteTool(),
        'yahoo': YahooMarketAPI()
        # 'alpha_vantage': AlphaVantageAPI(),
        # 'twitter': TwitterAPI(),
        # 'news': NewsAPI(),
        # 'federal': FederalReserveAPI()
    }
    
    # Validate all tools at once
    for name, tool in tools.items():
        if not all(hasattr(tool, attr) for attr in ['name', 'description']):
            raise ValueError(f"Tool {name} missing required attributes.")
    
    # Return all tools as a list for agent consumption
    return list(tools.values())

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

# Create agent factory to avoid repetition
def create_agent(role, goal, backstory, tools):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        allow_delegation=True,
        tools=tools
    )

# Create task factory to avoid repetition
def create_task(description, expected_output, agent):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )

def build_financial_crew(asset_selection="AAPL", risk_tolerance="Medium", news_impact_consideration=True):
    # Initialize tools once
    all_tools = initialize_tools()
    
    # Create agent definitions with reusable parameters
    agent_configs = {
        "market_data": {
            "role": "Market Data Agent",
            "goal": (
                "Fetch, aggregate, and analyze real-time and historical market data, "
                "including OHLCV (Open, High, Low, Close, Volume), technical indicators, "
                "news sentiment, and cryptocurrency market trends to assist traders and investors."
            ),
            "backstory": (
                "This agent is designed to provide comprehensive market intelligence by gathering financial data from "
                "multiple sources. It integrates with Yahoo Finance, Alpha Vantage, TradingView, and Binance to collect "
                "OHLCV data, fundamental indicators, and crypto trends. Additionally, it leverages news analysis and "
                "social media sentiment tracking to identify market-moving events, offering actionable insights for investors."
            ),
        },
        "sentiment_analysis": {
            "role": "Sentiment Analysis Agent",
            "goal": (
                "Monitor and analyze market sentiment by processing news, social media discussions, "
                "and financial reports to detect trends, assess investment risks, and predict market movements."
            ),
            "backstory": (
                "This agent specializes in financial sentiment analysis by aggregating and analyzing data "
                "from various sources, including real-time news, social media, and economic reports. "
                "By leveraging News APIs, X (formerly Twitter), Yahoo Finance, and Bloomberg, it classifies "
                "market sentiment as Bullish, Bearish, or Neutral, offering traders and investors valuable insights "
                "to refine their decision-making."
            ),
        },
        "macro_economic": {
            "role": "Macro-Economic Agent",
            "goal": (
                "Track and analyze macroeconomic indicators, including interest rates, inflation, "
                "GDP growth, and global economic policies, to predict market impact and guide "
                "investment strategies with defensive asset recommendations during downturns."
            ),
            "backstory": (
                "This agent specializes in macroeconomic analysis, aggregating and interpreting key policy data, "
                "global economic trends, and financial stability indicators. By leveraging real-time data from "
                "the Federal Reserve API, World Bank API, IMF, and market intelligence sources, it provides "
                "actionable insights to help investors anticipate risks, optimize asset allocation, and refine "
                "portfolio strategies in response to economic fluctuations."
            ),
        },
        "forex": {
            "role": "Forex Agent",
            "goal": (
                "Analyze foreign exchange markets, track currency fluctuations, and provide forex "
                "risk management strategies while identifying optimal international investment opportunities."
            ),
            "backstory": (
                "This agent specializes in forex market analysis, monitoring exchange rates and currency trends "
                "from multiple global sources. By integrating real-time data from the OANDA API, XE API, and financial "
                "market feeds, it assesses currency volatility, geopolitical risks, and macroeconomic factors affecting forex. "
                "It assists investors, traders, and businesses in hedging forex risk and making data-driven currency investment decisions."
            ),
        },
        "technical_analysis": {
            "role": "Technical Analysis Agent",
            "goal": (
                "Perform stock market analysis using technical indicators, fundamental valuation metrics, "
                "and stock screening tools to generate confidence scores for investments."
            ),
            "backstory": (
                "This agent specializes in technical stock analysis by computing key indicators such as "
                "RSI, MACD, Moving Averages, and Bollinger Bands. It also performs fundamental valuation using "
                "metrics like P/E ratio, EPS, and Book Value. By leveraging real-time data from TradingView API, "
                "Yahoo Finance API, and Screener.in API, it identifies support/resistance levels, trend patterns, "
                "and overall investment confidence scores. Additionally, it integrates financial news and macroeconomic "
                "indicators to refine market insights and improve investment decisions."
            ),
        },
        "portfolio_strategy": {
            "role": "Portfolio Strategy Agent",
            "goal": (
                "Develop and optimize trading strategies by analyzing historical data and applying portfolio models "
                "to maximize returns while minimizing risk."
            ),
            "backstory": (
                "This agent specializes in portfolio strategy by leveraging historical market data and quantitative models. "
                "It suggests trading strategies such as momentum, mean reversion, and value investing. "
                "Additionally, it optimizes investment portfolios using Modern Portfolio Theory (MPT) to achieve maximum returns "
                "with minimal risk. By integrating data from QuantConnect API and MPT models, it computes expected returns based on user input."
            ),
        },
        "risk_management": {
            "role": "Advanced Risk Advisor",
            "goal": (
                "Conduct real-time and historical risk analysis using financial models, "
                "news sentiment, market volatility, and economic indicators. "
                "Evaluate risk exposure, predict potential threats, and provide risk-adjusted strategies "
                "to ensure alignment with the firm's risk tolerance and investment objectives."
            ),
            "backstory": (
                "Equipped with expertise in risk modeling, quantitative finance, and market analysis, "
                "this agent evaluates financial risks using Monte Carlo simulations, Value at Risk (VaR), "
                "and stress testing techniques. It continuously scans market data, central bank policies, "
                "and news sentiment to assess risk levels. "
                "The agent also monitors economic indicators such as inflation, interest rates, "
                "and GDP growth to provide a holistic risk outlook."
            ),
        }
    }
    
    # Task descriptions with placeholders for dynamic content
    task_configs = {
        "market_data": {
            "description": (
                "Gather, analyze, and report real-time and historical market data "
                f"for the selected asset ({asset_selection}). "
                "Leverage technical indicators, fundamental data, and sentiment analysis "
                "to provide actionable market insights."
            ),
            "expected_output": (
                f"A comprehensive report on {asset_selection}, including OHLCV trends, "
                "technical indicators, market sentiment, and potential trading opportunities."
            ),
        },
        "sentiment_analysis": {
            "description": (
                f"Continuously monitor and analyze market sentiment for the selected asset ({asset_selection}). "
                "Leverage real-time news, social media discussions, and financial reports to classify sentiment "
                "as Bullish, Bearish, or Neutral."
            ),
            "expected_output": (
                f"A detailed sentiment analysis report on {asset_selection}, including aggregated news sentiment, "
                "social media sentiment trends, and an assessment of potential market impact."
            ),
        },
        "macro_economic": {
            "description": (
                "Monitor and analyze macroeconomic indicators such as interest rates, inflation, GDP growth, and "
                "global economic policies. Assess their potential impact on financial markets and investment strategies."
            ),
            "expected_output": (
                "A comprehensive macroeconomic report covering key indicators, policy shifts, and market impact analysis. "
                "Includes defensive asset recommendations for potential downturns."
            ),
        },
        "forex_analysis": {
            "description": (
                "Continuously monitor and analyze forex market trends, currency fluctuations, and macroeconomic factors "
                "affecting foreign exchange rates. Assess geopolitical risks and provide forex risk management strategies "
                f"for the selected asset ({asset_selection})."
            ),
            "expected_output": (
                "A detailed forex market report with currency trend analysis, volatility assessment, "
                "and optimal trading or hedging recommendations."
            ),
        },
        "technical_analysis": {
            "description": (
                f"Analyze stock market trends for {asset_selection} using technical indicators (RSI, MACD, Moving Averages, "
                "Bollinger Bands) and fundamental valuation metrics (P/E ratio, EPS, Book Value). Identify support/resistance "
                "levels, trend patterns, and generate confidence scores."
            ),
            "expected_output": (
                f"A comprehensive technical analysis report for {asset_selection} with stock trend predictions, "
                "support/resistance insights, and confidence scores."
            ),
        },
        "portfolio_strategy": {
            "description": (
                f"Analyze historical market data for {asset_selection} and apply portfolio optimization models "
                "to develop trading strategies. Evaluate strategies like momentum trading, mean reversion, and value investing. "
                f"Optimize portfolio allocation using Modern Portfolio Theory (MPT) while considering risk tolerance ({risk_tolerance})."
            ),
            "expected_output": (
                "A portfolio strategy report detailing optimal asset allocation, expected returns, risk assessment, "
                "and recommended trading strategies."
            ),
        },
        "risk_management": {
            "description": (
                f"Conduct real-time and historical risk assessments for {asset_selection} using Monte Carlo simulations, "
                "Value at Risk (VaR), and stress testing techniques. Analyze market volatility, macroeconomic indicators, "
                f"and news sentiment (News Impact Consideration: {news_impact_consideration}) to evaluate risk exposure."
            ),
            "expected_output": (
                "A comprehensive risk assessment report detailing financial threats, VaR calculations, stress test results, "
                "and risk-adjusted recommendations aligned with investment objectives."
            ),
        }
    }
    
    # Create agents using the factory method
    agents = {}
    for key, config in agent_configs.items():
        agents[key] = create_agent(
            role=config["role"],
            goal=config["goal"],
            backstory=config["backstory"],
            tools=all_tools
        )
    
    # Create tasks using the factory method
    tasks = []
    for key, config in task_configs.items():
        # Map task to corresponding agent (forex_analysis task uses forex agent)
        agent_key = key if key != "forex_analysis" else "forex"
        tasks.append(create_task(
            description=config["description"],
            expected_output=config["expected_output"],
            agent=agents[agent_key]
        ))
    
    # Create manager LLM
    manager_llm = CustomLLM()
    
    # Create the Crew
    financial_crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        manager_llm=manager_llm,
        process=Process.hierarchical,
        verbose=True
    )
    
    return financial_crew

def run_financial_analysis(asset_selection="AAPL", initial_capital="100000", 
                          risk_tolerance="Medium", trading_strategy="Day Trading", 
                          news_impact_consideration=True):
    # Build the crew with specific parameters
    crew = build_financial_crew(
        asset_selection=asset_selection,
        risk_tolerance=risk_tolerance,
        news_impact_consideration=news_impact_consideration
    )
    
    # Prepare inputs
    financial_trading_inputs = {
        'asset_selection': asset_selection,
        'initial_capital': initial_capital,
        'risk_tolerance': risk_tolerance,
        'trading_strategy_preference': trading_strategy,
        'news_impact_consideration': news_impact_consideration
    }
    
    # Run the analysis
    result = crew.kickoff(inputs=financial_trading_inputs)
    return result

# Example usage
if __name__ == "__main__":
    result = run_financial_analysis()
    print(result)