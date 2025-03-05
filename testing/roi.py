# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew, Process
import os
from utils import get_serper_key
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
import re

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
        'yahoo': YahooMarketAPI(),
        'alpha_vantage': AlphaVantageAPI(),
        'twitter': TwitterAPI(),
        'news': NewsAPI(),
        'federal': FederalReserveAPI()
    }
    
    # Validate all tools at once
    for name, tool in tools.items():
        if not all(hasattr(tool, attr) for attr in ['name', 'description']):
            raise ValueError(f"Tool {name} missing required attributes.")
    
    # Return all tools as a list for agent consumption
    return list(tools.values())

# Create a custom LLM class
# class CustomLLM(LLM):
#     model_name: str = "ollama/llama2"
#     temperature: float = 0.7
#     max_tokens: int = 1024
    
#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         try:
#             messages = [{"role": "user", "content": prompt}]
#             response = completion(
#                 model=self.model_name,
#                 messages=messages,
#                 temperature=self.temperature,
#                 max_tokens=self.max_tokens,
#                 stop=stop
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"Error processing with {self.model_name}: {str(e)}"
    
#     @property
#     def _llm_type(self) -> str:
#         return "custom_llm"
    
#     @property
#     def _identifying_params(self) -> Mapping[str, Any]:
#         return {"model_name": self.model_name, "temperature": self.temperature}
    

class LiteLLMWrapper(LLM):
    model_name: str = "gemini/gemini-2.0-flash"
    temperature: float = 0.7
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            stop=stop
        )
        return response.choices[0].message.content
    
    @property
    def _llm_type(self) -> str:
        return "litellm"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

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
                "Include clear calculations on expected returns based on the initial investment amount."
            ),
            "expected_output": (
                "A portfolio strategy report detailing optimal asset allocation, expected returns, risk assessment, "
                "and recommended trading strategies with explicit ROI projections for different time horizons."
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
    # manager_llm = CustomLLM() 
    manager_llm = LiteLLMWrapper()
    
    # Create the Crew
    financial_crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        manager_llm=manager_llm,
        process=Process.hierarchical,
        verbose=True
    )
    
    return financial_crew

def extract_roi_data(analysis_result, initial_capital):
    """Extract ROI data from the analysis result"""
    # Convert initial capital to float
    try:
        initial_capital = float(initial_capital)
    except ValueError:
        initial_capital = float(initial_capital.replace(',', ''))
    
    # Regular expressions to find ROI or return percentages
    roi_patterns = [
        r"expected\s+return(?:\s+of)?\s*(?:is|:|\s)?\s*[-+]?\d+(?:\.\d+)?%",
        r"projected\s+return(?:\s+of)?\s*(?:is|:|\s)?\s*[-+]?\d+(?:\.\d+)?%",
        r"ROI\s*(?:is|:|\s)?\s*[-+]?\d+(?:\.\d+)?%",
        r"return\s+on\s+investment\s*(?:is|:|\s)?\s*[-+]?\d+(?:\.\d+)?%",
        r"(?:expected|projected|estimated|predicted|anticipated)\s+(?:to\s+)?(?:yield|return|gain|appreciate)\s+(?:by|at)?\s*[-+]?\d+(?:\.\d+)?%"
    ]
    
    # Search for ROI mentions
    roi_percentages = []
    for pattern in roi_patterns:
        matches = re.findall(pattern, analysis_result, re.IGNORECASE)
        for match in matches:
            # Extract the percentage value
            percentage = re.search(r"[-+]?\d+(?:\.\d+)?%", match)
            if percentage:
                roi_percentages.append(percentage.group(0))
    
    # Also look for time-based returns
    time_patterns = {
        "annual": r"(?:annual|yearly|per\s+year)(?:\s+return|\s+ROI|\s+yield)?\s*(?:of|:)?\s*[-+]?\d+(?:\.\d+)?%",
        "monthly": r"(?:monthly|per\s+month)(?:\s+return|\s+ROI|\s+yield)?\s*(?:of|:)?\s*[-+]?\d+(?:\.\d+)?%",
        "quarterly": r"(?:quarterly|per\s+quarter)(?:\s+return|\s+ROI|\s+yield)?\s*(?:of|:)?\s*[-+]?\d+(?:\.\d+)?%"
    }
    
    time_based_returns = {}
    for time_period, pattern in time_patterns.items():
        matches = re.findall(pattern, analysis_result, re.IGNORECASE)
        for match in matches:
            percentage = re.search(r"[-+]?\d+(?:\.\d+)?%", match)
            if percentage:
                time_based_returns[time_period] = percentage.group(0)
    
    # Calculate projected returns in currency values
    roi_values = {}
    
    # Process general ROI
    if roi_percentages:
        # Use the first found percentage as primary ROI
        primary_roi = roi_percentages[0]
        percentage_value = float(primary_roi.strip('%').replace('+', ''))
        roi_values["general"] = {
            "percentage": primary_roi,
            "value": round(initial_capital * (percentage_value / 100), 2)
        }
    
    # Process time-based returns
    for period, percentage in time_based_returns.items():
        percentage_value = float(percentage.strip('%').replace('+', ''))
        roi_values[period] = {
            "percentage": percentage,
            "value": round(initial_capital * (percentage_value / 100), 2)
        }
    
    # If no ROI information was found, provide a default estimation
    if not roi_values:
        roi_values["estimated"] = {
            "percentage": "8-12%",
            "value": f"{round(initial_capital * 0.08, 2)} - {round(initial_capital * 0.12, 2)} (estimated)"
        }
    
    return roi_values

def format_report(analysis_result, asset_selection, initial_capital, risk_tolerance, trading_strategy, roi_data):
    """Format the analysis result into a pretty report"""
    console = Console()
    
    # Create header
    header = Panel(
        f"[bold cyan]Financial Analysis Report: {asset_selection}[/bold cyan]",
        expand=False
    )
    
    # Create investment details table
    investment_table = Table(title="Investment Parameters")
    investment_table.add_column("Parameter", style="cyan")
    investment_table.add_column("Value", style="green")
    
    investment_table.add_row("Asset", asset_selection)
    investment_table.add_row("Initial Investment", f"${initial_capital}")
    investment_table.add_row("Risk Tolerance", risk_tolerance)
    investment_table.add_row("Trading Strategy", trading_strategy)
    
    # Create ROI table
    roi_table = Table(title="Expected Returns")
    roi_table.add_column("Time Horizon", style="cyan")
    roi_table.add_column("Percentage", style="green")
    roi_table.add_column("Value", style="green")
    
    for period, data in roi_data.items():
        period_display = period.capitalize()
        roi_table.add_row(period_display, data["percentage"], f"${data['value']}")
    
    # Format the analysis text with markdown
    # Clean up the analysis text
    cleaned_analysis = analysis_result
    
    # If analysis is extremely long, truncate it
    if len(cleaned_analysis) > 5000:
        cleaned_analysis = cleaned_analysis[:5000] + "...\n[Analysis truncated for readability]"
    
    analysis_md = Markdown(cleaned_analysis)
    
    # Return all components for display
    return {
        "header": header,
        "investment_table": investment_table,
        "roi_table": roi_table,
        "analysis": analysis_md
    }

def prompt_for_inputs():
    """Prompt for user inputs for financial analysis"""
    console = Console()
    
    console.print("\n[bold cyan]===== Investment Analysis Tool =====[/bold cyan]")
    console.print("[italic]Please provide the following investment parameters:[/italic]\n")
    
    # Get asset selection
    asset_selection = console.input("[bold green]Asset Symbol/Ticker[/bold green] (e.g., AAPL, MSFT, BTC): ")
    
    # Get initial capital
    while True:
        initial_capital = console.input("[bold green]Initial Investment Amount[/bold green] ($): ")
        try:
            # Remove commas if present
            test_capital = float(initial_capital.replace(',', ''))
            break
        except ValueError:
            console.print("[bold red]Please enter a valid number.[/bold red]")
    
    # Get risk tolerance
    risk_options = ["Low", "Medium", "High"]
    console.print("[bold green]Risk Tolerance:[/bold green]")
    for i, option in enumerate(risk_options, 1):
        console.print(f"  {i}. {option}")
    
    while True:
        risk_choice = console.input("Select (1-3): ")
        try:
            risk_index = int(risk_choice) - 1
            if 0 <= risk_index < len(risk_options):
                risk_tolerance = risk_options[risk_index]
                break
            else:
                console.print("[bold red]Please select a valid option.[/bold red]")
        except ValueError:
            console.print("[bold red]Please enter a number.[/bold red]")
    
    # Get trading strategy
    strategy_options = ["Day Trading", "Swing Trading", "Position Trading", "Long-term Investment"]
    console.print("[bold green]Trading Strategy:[/bold green]")
    for i, option in enumerate(strategy_options, 1):
        console.print(f"  {i}. {option}")
    
    while True:
        strategy_choice = console.input("Select (1-4): ")
        try:
            strategy_index = int(strategy_choice) - 1
            if 0 <= strategy_index < len(strategy_options):
                trading_strategy = strategy_options[strategy_index]
                break
            else:
                console.print("[bold red]Please select a valid option.[/bold red]")
        except ValueError:
            console.print("[bold red]Please enter a number.[/bold red]")
    
    # Get news impact consideration
    while True:
        news_choice = console.input("[bold green]Consider News Impact[/bold green] (y/n): ").lower()
        if news_choice in ['y', 'yes']:
            news_impact_consideration = True
            break
        elif news_choice in ['n', 'no']:
            news_impact_consideration = False
            break
        else:
            console.print("[bold red]Please enter 'y' or 'n'.[/bold red]")
    
    return {
        "asset_selection": asset_selection,
        "initial_capital": initial_capital,
        "risk_tolerance": risk_tolerance,
        "trading_strategy": trading_strategy,
        "news_impact_consideration": news_impact_consideration
    }

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

def main():
    """Main function to run the investment analysis tool with formatted output"""
    console = Console()
    
    # Welcome message
    console.print("\n[bold cyan]Welcome to the Financial Investment Analysis Tool[/bold cyan]")
    console.print("[italic]This tool provides comprehensive market analysis and investment projections[/italic]\n")
    
    # Get user inputs
    inputs = prompt_for_inputs()
    
    # Show processing message
    with console.status("[bold green]Processing your investment analysis...[/bold green]", spinner="dots"):
        # Run the analysis
        analysis_result = run_financial_analysis(
            asset_selection=inputs["asset_selection"],
            initial_capital=inputs["initial_capital"],
            risk_tolerance=inputs["risk_tolerance"],
            trading_strategy=inputs["trading_strategy"],
            news_impact_consideration=inputs["news_impact_consideration"]
        )
    
    # Extract ROI data
    roi_data = extract_roi_data(analysis_result, inputs["initial_capital"])
    
    # Format the report
    report = format_report(
        analysis_result=analysis_result,
        asset_selection=inputs["asset_selection"],
        initial_capital=inputs["initial_capital"],
        risk_tolerance=inputs["risk_tolerance"],
        trading_strategy=inputs["trading_strategy"],
        roi_data=roi_data
    )
    
    # Display the report
    console.print("\n\n")
    console.print(report["header"])
    console.print(report["investment_table"])
    console.print(report["roi_table"])
    console.print("\n[bold cyan]Analysis Details:[/bold cyan]")
    console.print(report["analysis"])
    
    # Save report option
    save_option = console.input("\n[bold green]Would you like to save this report? (y/n): [/bold green]").lower()
    if save_option in ['y', 'yes']:
        filename = f"financial_analysis_{inputs['asset_selection']}_{inputs['trading_strategy'].replace(' ', '_')}.txt"
        with open(filename, "w") as f:
            f.write(f"FINANCIAL ANALYSIS REPORT: {inputs['asset_selection']}\n\n")
            f.write(f"INVESTMENT PARAMETERS:\n")
            f.write(f"Asset: {inputs['asset_selection']}\n")
            f.write(f"Initial Investment: ${inputs['initial_capital']}\n")
            f.write(f"Risk Tolerance: {inputs['risk_tolerance']}\n")
            f.write(f"Trading Strategy: {inputs['trading_strategy']}\n\n")
            
            f.write(f"EXPECTED RETURNS:\n")
            for period, data in roi_data.items():
                f.write(f"{period.capitalize()}: {data['percentage']} (${data['value']})\n")
            
            f.write(f"\nANALYSIS DETAILS:\n{analysis_result}")
        
        console.print(f"[bold green]Report saved to {filename}[/bold green]")
    
    return analysis_result, roi_data

# Example usage
if __name__ == "__main__":
    analysis_result, roi_data = main()