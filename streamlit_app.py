import streamlit as st
from main_financial_advisor import run_financial_advisor, LiteLLMWrapper
import os

st.title("Financial Investment Analysis Tool")

# Gemini API Key
gemini_api_key = st.text_input("Gemini API Key", type="password")
os.environ["GEMINI_API_KEY"] = gemini_api_key

# Risk Tolerance Questionnaire
st.subheader("Risk Tolerance Assessment")
q1 = st.selectbox("What are your most important financial goals?", ["Retirement", "Capital Preservation", "Growth"])
q2 = st.selectbox("How would you describe your knowledge of investments?", ["Limited", "Some Knowledge", "Experienced"])
q3 = st.selectbox("How would you react to a significant decline in the value of your investments?", ["Sell to avoid further losses", "Hold steady", "Buy more"])
q4 = st.selectbox("Over what time period do you expect to achieve your financial goals?", ["Less than 5 years", "5-10 years", "More than 10 years"])
q5 = st.selectbox("Which of the following statements best describes your attitude toward taking investment risks?", ["I am not willing to take any risks", "I am willing to take some risks for a higher potential return", "I am comfortable taking significant risks for a potentially high return"])

# User Input
asset_selection = st.text_input("Asset Symbol/Ticker (e.g., AAPL, MSFT, BTC)", "AAPL")
initial_investment = st.number_input("Initial Investment Amount ($)", value=10000.0)
trading_strategy = st.selectbox(
    "Trading Strategy",
    options=["Day Trading", "Swing Trading", "Position Trading", "Long-term Investment"]
)
news_impact = st.checkbox("Consider News Impact", value=True)

if st.button("Run Analysis"):
    if not gemini_api_key:
        st.error("Please enter your Gemini API Key.")
    else:
        # Map risk tolerance responses to a single value
        risk_tolerance_mapping = {
            "I am not willing to take any risks": "Low",
            "I am willing to take some risks for a higher potential return": "Medium",
            "I am comfortable taking significant risks for a potentially high return": "High"
        }
        risk_tolerance = risk_tolerance_mapping[q5]

        # Run Financial Advisor
        result = run_financial_advisor(
            initial_investment=initial_investment,
            risk_tolerance=risk_tolerance,
            trading_strategy=trading_strategy,
            asset_selection=asset_selection,
            news_impact=news_impact
        )

        # Display Results
        st.subheader("Financial Advisory Report")

        # Pretty Print the Results
        try:
            report = eval(result)  # Evaluate the string as a Python dictionary
            st.write("### Investment Strategy:")
            st.write(report['tasks_output'][2]['raw'])

            # Text-to-Speech
            try:
                from gtts import gTTS
                import io
                tts = gTTS(text=report['tasks_output'][2]['raw'], lang='en')
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                st.audio(mp3_fp, format="audio/mp3")
            except ImportError:
                st.warning("gTTS library not found. Please install it to enable text-to-speech: pip install gTTS")
            except Exception as e:
                st.error(f"Error generating text-to-speech: {e}")

        except Exception as e:
            st.error(f"Error processing the report: {e}")
