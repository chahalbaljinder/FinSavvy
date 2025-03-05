import warnings
import logging
import os
import pprint
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from tools import YahooMarketAPI
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from litellm import completion

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

api_key = os.getenv("GEMINI_API_KEY")

def initialize_tools():
    tools = {
        'search': SerperDevTool(),
        'scrape': ScrapeWebsiteTool(),
        'yahoo': YahooMarketAPI()
    }
    
    for name, tool in tools.items():
        if not all(hasattr(tool, attr) for attr in ['name', 'description']):
            raise ValueError(f"Tool {name} is missing required attributes.")
    
    logging.info("Tools initialized successfully.")
    return list(tools.values())

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

def create_agent(role, goal, backstory, tools):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        allow_delegation=True,
        tools=tools
    )

def create_task(description, expected_output, agent):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )

def build_financial_crew(initial_investment, risk_tolerance, trading_strategy, asset_selection, news_impact):
    logging.info("Building financial crew...")
    
    all_tools = initialize_tools()

    agents = {
        "market_data": create_agent(
            "Market Data Analyst",
            "Gather and analyze market data to provide investment insights.",
            "Specializes in financial market analysis using real-time data.",
            all_tools
        ),
        "risk_management": create_agent(
            "Risk Management Analyst",
            "Evaluate risk factors and suggest mitigation strategies.",
            "Expert in risk analysis and market fluctuations.",
            all_tools
        ),
        "investment_strategy": create_agent(
            "Investment Strategy Advisor",
            "Develop optimal investment strategies based on user preferences.",
            "Analyzes trading strategies to provide optimal investment plans.",
            all_tools
        )
    }

    tasks = [
        create_task(
            f"Analyze market trends for {asset_selection} investments.",
            "Detailed market trend analysis and investment suggestions.",
            agents["market_data"]
        ),
        create_task(
            "Assess portfolio risk based on market conditions and user preferences.",
            "Risk assessment report with mitigation strategies.",
            agents["risk_management"]
        ),
        create_task(
            "Suggest an investment strategy aligned with user risk tolerance and trading preferences.",
            "Personalized investment strategy report.",
            agents["investment_strategy"]
        )
    ]
    
    manager_llm = LiteLLMWrapper(api_key=api_key)
    
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        manager_llm=manager_llm,
        process=Process.hierarchical,
        verbose=True
    )
    
    logging.info("Crew and tasks successfully built.")
    return crew

def run_financial_advisor(initial_investment, risk_tolerance, trading_strategy, asset_selection, news_impact):
    crew = build_financial_crew(initial_investment, risk_tolerance, trading_strategy, asset_selection, news_impact)
    inputs = {
        "initial_investment": initial_investment,
        "risk_tolerance": risk_tolerance,
        "trading_strategy": trading_strategy,
        "asset_selection": asset_selection,
        "news_impact": news_impact
    }
    logging.info("Starting financial advisory process...")
    result = crew.kickoff(inputs=inputs)
    
    print("\n=== Financial Advisory Report ===")
    pprint.pprint(result, width=100)
    print("\n=== End of Report ===")
    return result

if __name__ == "__main__":
    print("\nWelcome to the Financial Investment Analysis Tool")
    print("This tool provides comprehensive market analysis and investment projections\n")
    print("===== Investment Analysis Tool =====")
    print("Please provide the following investment parameters:\n")
    
    asset_selection = input("\033[1mAsset Symbol/Ticker\033[0m (e.g., AAPL, MSFT, BTC): ")
    initial_investment = float(input("\033[1mInitial Investment Amount ($):\033[0m "))
    
    print("\033[1mRisk Tolerance:\033[0m")
    print("1. Low\n2. Medium\n3. High")
    risk_tolerance = input("Select (1-3): ")
    
    print("\033[1mTrading Strategy:\033[0m")
    print("1. Day Trading\n2. Swing Trading\n3. Position Trading\n4. Long-term Investment")
    trading_strategy = input("Select (1-4): ")
    
    news_impact = input("\033[1mConsider News Impact (y/n):\033[0m ").lower() == 'y'
    
    run_financial_advisor(initial_investment, risk_tolerance, trading_strategy, asset_selection, news_impact)