import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# 1. Setup the AI
@st.cache_resource
def load_model():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

sia = load_model() 
# 2. Build the Website Interface
st.set_page_config(page_title="AI Sentiment Analyser")
st.title(" AI Sentiment Analyser")
st.write("Enter text below to let the AI detect the emotion.")
user_input = st.text_area("Your Message:", placeholder="Type here...")

if st.button("Analyze"):
    if user_input:
        score = sia.polarity_scores(user_input)['compound']
        if score >= 0.05:
            st.success(f"Positive  (Score: {score})")
        elif score <= -0.05:
            st.error(f"Negative  (Score: {score})")
        else:
            st.info(f"Neutral  (Score: {score})")
    else:
        st.warning("Please enter text.") 


        