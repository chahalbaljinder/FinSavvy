import streamlit as st
from main_financial_advisor import run_financial_advisor

st.title("Financial Investment Analysis Tool")

asset_selection = st.text_input("Asset Symbol/Ticker (e.g., AAPL, MSFT, BTC)", "AAPL")
initial_investment = st.number_input("Initial Investment Amount ($)", value=10000.0)

risk_tolerance = st.selectbox(
    "Risk Tolerance",
    options=["Low", "Medium", "High"]
)

trading_strategy = st.selectbox(
    "Trading Strategy",
    options=["Day Trading", "Swing Trading", "Position Trading", "Long-term Investment"]
)

news_impact = st.checkbox("Consider News Impact", value=True)

if st.button("Run Analysis"):
    result = run_financial_advisor(
        initial_investment=initial_investment,
        risk_tolerance=risk_tolerance,
        trading_strategy=trading_strategy,
        asset_selection=asset_selection,
        news_impact=news_impact
    )
    st.write("### Financial Advisory Report")
    st.write(result)
