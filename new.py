import warnings
import logging
from crewai import Agent, Task, Crew, Process
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from tools import YahooMarketAPI, AlphaVantageAPI, TwitterAPI, NewsAPI, FederalReserveAPI
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from litellm import completion

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

api_key = os.getenv("GEMINI_API_KEY")

# Initialize tools once, optimize and check their validity
def initialize_tools():
    tools = {
        'search': SerperDevTool(),
        'scrape': ScrapeWebsiteTool(),
        'yahoo': YahooMarketAPI()
    }
    
    # Validate tools
    for name, tool in tools.items():
        if not all(hasattr(tool, attr) for attr in ['name', 'description']):
            raise ValueError(f"Tool {name} is missing required attributes.")
    
    logging.info("Tools initialized successfully.")
    return list(tools.values())

################################################################################################

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
    
################################################################################################


# Factory for creating agents to avoid repetition
def create_agent(role, goal, backstory, tools):
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=True,
        allow_delegation=True,
        tools=tools
    )

# Factory for creating tasks to avoid repetition
def create_task(description, expected_output, agent):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )

# Build the financial crew and tasks
def build_financial_crew(initial_investment, risk_tolerance, trading_strategy, market_type):
    logging.info(f"Building financial crew for market type {market_type} with risk tolerance {risk_tolerance}.")
    
    all_tools = initialize_tools()

    # Define agents
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

    # Define tasks
    tasks = [
        create_task(
            f"Analyze market trends and provide insights for {market_type} investments.",
            "Detailed market trend analysis and investment suggestions.",
            agents["market_data"]
        ),
        create_task(
            "Assess portfolio risk based on market conditions and user preferences.",
            "Risk assessment report with mitigation strategies.",
            agents["risk_management"]
        ),
        create_task(
            "Suggest an investment strategy that aligns with user risk tolerance and trading preferences.",
            "Personalized investment strategy report.",
            agents["investment_strategy"]
        )
    ]
    
    # Create manager LLM
    #manager_llm = CustomLLM() # for llama
    manager_llm = LiteLLMWrapper(api_key = api_key)
    
    # Create Crew
    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        manager_llm=manager_llm,
        process=Process.hierarchical,
        verbose=True
    )

    logging.info("Crew and tasks successfully built.")
    return crew

# Run the financial advisor system
def run_financial_advisor(initial_investment, risk_tolerance, trading_strategy, market_type):
    crew = build_financial_crew(initial_investment, risk_tolerance, trading_strategy, market_type)
    inputs = {
        "initial_investment": initial_investment,
        "risk_tolerance": risk_tolerance,
        "trading_strategy": trading_strategy,
        "market_type": market_type
    }
    logging.info("Starting financial advisory process...")
    result = crew.kickoff(inputs=inputs)
    return result

# Example usage
if __name__ == "__main__":
    run_financial_advisor(10000, "Medium", "Growth Investing", "Stocks")
