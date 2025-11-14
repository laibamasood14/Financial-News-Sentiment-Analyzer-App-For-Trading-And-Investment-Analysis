
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

import requests

@st.cache_data(ttl=3600)
def fetch_latest_financial_news(limit=10):
    API_KEY = "NewsData_API_Key"
    url = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q=stocks%20OR%20finance%20OR%20markets&language=en&category=business"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            news_items = []
            for item in data["results"][:limit]:
                news_items.append({
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "source": item.get("source_id"),
                    "pubDate": item.get("pubDate"),
                    "link": item.get("link")
                })
            return pd.DataFrame(news_items)
        else:
            st.warning("No news found from API.")
            return pd.DataFrame()
    else:
        st.error(f"API error: {response.status_code}")
        return pd.DataFrame()

# Page configuration
st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #6c757d;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Text Preprocessor Class
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        financial_keeps = {'up', 'down', 'above', 'below', 'over', 'under'}
        self.stop_words = self.stop_words - financial_keeps
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', '', text)
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) 
                     for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
            return ' '.join(tokens)
        except:
            return text

# Load Model
@st.cache_resource
def load_model():
    try:
        model_path = "./enhanced_finbert_model"  # Path to your fine-tuned model folder
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize
preprocessor = TextPreprocessor()
tokenizer, model = load_model()

# Prediction Function
def predict_sentiment(text):
    if not text.strip():
        return None
    
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=128, padding=True)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        
        return {
            'sentiment': sentiment_map[prediction],
            'confidence': confidence,
            'probabilities': {
                'Negative': float(probs[0][0]),
                'Neutral': float(probs[0][1]),
                'Positive': float(probs[0][2])
            }
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Visualization Functions
def create_gauge_chart(confidence, sentiment):
    colors = {
        'Positive': '#28a745',
        'Neutral': '#ffc107',
        'Negative': '#dc3545'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {sentiment}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': colors[sentiment]},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_probability_chart(probabilities):
    colors = ['#dc3545', '#ffc107', '#28a745']
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities.keys()),
            y=[p * 100 for p in probabilities.values()],
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Sentiment Probability Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False
    )
    
    return fig

# Sidebar
with st.sidebar:
    #st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Financial+AI", 
             #use_container_width=True)
    
    st.title("📊 Navigation")
    page = st.radio("Select Page:", 
                    ["Single Analysis", "Batch Analysis","Live Financial News", "About Project"])
    
    st.markdown("---")
    
    st.subheader("ℹ️ About")
    st.markdown("""
    This application analyzes sentiment in financial news and reports using 
    **FinBERT**, a BERT model trained on financial text.
    
    **Sentiment Classes:**
    - 🟢 Positive
    - 🟡 Neutral
    - 🔴 Negative
    """)
    
    st.markdown("---")
    
    st.subheader("👥 Team")
    st.markdown("""
    **Group CS-A**
    - Laiba Masood (CT-22001)
    - Aiza Asim (CT-22006)
    - Zainab Fatima (CT-22007)
    - Maryam Shaikh (CT-22012)
    
    **Course:** CT-485 NLP  
    **Institution:** NED University
    """)

# Main Content
if page == "Single Analysis":
    st.title("📈 Financial Sentiment Analyzer")
    st.markdown("Analyze sentiment in financial news, reports, and market commentary")
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Financial Text")
        text_input = st.text_area(
            "Type or paste financial text here:",
            height=150,
            placeholder="Example: The company's quarterly earnings exceeded expectations, showing strong revenue growth..."
        )
        
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
    
    with col2:
        st.subheader("Quick Examples")
        examples = {
            "Positive": "Earnings per share grew 15% year-over-year, signaling robust financial performance.",
            "Negative": "Unexpected operational issues caused production delays, affecting revenue.",
            "Neutral": "No significant corporate announcements were made regarding mergers or acquisitions."
        }
        
        for sentiment, text in examples.items():
            if st.button(f"Try {sentiment} Example"):
                text_input = text
                analyze_button = True
    
    # Analysis
    if analyze_button and text_input:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(text_input)
            
            if result:
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Metrics Row
                col1, col2, col3 = st.columns(3)
                
                sentiment_class = result['sentiment'].lower()
                
                with col1:
                    st.metric(
                        "Sentiment",
                        result['sentiment'],
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Confidence",
                        f"{result['confidence']*100:.1f}%",
                        delta=None
                    )
                
                with col3:
                    interpretation = "Strong signal" if result['confidence'] > 0.8 else "Moderate signal"
                    st.metric(
                        "Interpretation",
                        interpretation,
                        delta=None
                    )
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(
                        create_gauge_chart(result['confidence'], result['sentiment']),
                        use_container_width=True
                    )
                
                with col2:
                    st.plotly_chart(
                        create_probability_chart(result['probabilities']),
                        use_container_width=True
                    )
                
                # Detailed Probabilities
                st.subheader("Detailed Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Sentiment': list(result['probabilities'].keys()),
                    'Probability': [f"{p*100:.2f}%" for p in result['probabilities'].values()],
                    'Raw Score': list(result['probabilities'].values())
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Text Analysis
                with st.expander("🔍 Text Analysis Details"):
                    cleaned = preprocessor.clean_text(text_input)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Text:**")
                        st.info(text_input)
                        st.markdown(f"**Length:** {len(text_input)} characters")
                    
                    with col2:
                        st.markdown("**Processed Text:**")
                        st.info(cleaned)
                        st.markdown(f"**Word Count:** {len(cleaned.split())} words")

elif page == "Batch Analysis":
    st.title("📁 Batch Sentiment Analysis")
    st.markdown("Analyze multiple financial texts at once")
    
    # Option 1: Upload CSV
    st.subheader("Option 1: Upload CSV File")
    uploaded_file = st.file_uploader(
        "Upload a CSV file with a 'text' column",
        type=['csv']
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column")
        else:
            st.success(f"Loaded {len(df)} texts")
            
            if st.button("Analyze All"):
                progress_bar = st.progress(0)
                results = []
                
                for idx, text in enumerate(df['text']):
                    result = predict_sentiment(str(text))
                    if result:
                        results.append({
                            'text': text[:100] + '...' if len(text) > 100 else text,
                            'sentiment': result['sentiment'],
                            'confidence': result['confidence']
                        })
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                
                # Summary Statistics
                st.subheader("📊 Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                sentiment_counts = results_df['sentiment'].value_counts()
                
                with col1:
                    st.metric("Positive", sentiment_counts.get('Positive', 0))
                with col2:
                    st.metric("Neutral", sentiment_counts.get('Neutral', 0))
                with col3:
                    st.metric("Negative", sentiment_counts.get('Negative', 0))
                
                # Visualization
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={
                        'Positive': '#28a745',
                        'Neutral': '#ffc107',
                        'Negative': '#dc3545'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results Table
                st.subheader("Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download Results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "sentiment_analysis_results.csv",
                    "text/csv"
                )
    
    # Option 2: Manual Entry
    st.markdown("---")
    st.subheader("Option 2: Manual Entry")
    
    num_texts = st.number_input("Number of texts to analyze:", 
                                min_value=1, max_value=10, value=3)
    
    texts = []
    for i in range(num_texts):
        text = st.text_input(f"Text {i+1}:", key=f"batch_text_{i}")
        if text:
            texts.append(text)
    
    if st.button("Analyze Texts") and texts:
        results = []
        for text in texts:
            result = predict_sentiment(text)
            if result:
                results.append({
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'sentiment': result['sentiment'],
                    'confidence': f"{result['confidence']*100:.1f}%"
                })
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
elif page == "Live Financial News":
    st.title("📰 Real-Time Financial News Sentiment")
    st.markdown("Fetching live news using NewsData.io API")

    num_articles = st.slider("Number of articles to analyze", 5, 20, 10)
    news_df = fetch_latest_financial_news(limit=num_articles)

    if not news_df.empty:
        st.success(f"Fetched {len(news_df)} recent news articles.")
        st.dataframe(news_df[["title", "source", "pubDate"]], use_container_width=True)

        if st.button("Analyze Live News Sentiment"):
            st.info("Analyzing live financial news...")
            results = []
            for _, row in news_df.iterrows():
                result = predict_sentiment(str(row["title"]))
                if result:
                    results.append({
                        "title": row["title"],
                        "sentiment": result["sentiment"],
                        "confidence": f"{result['confidence']*100:.2f}%"
                    })
            results_df = pd.DataFrame(results)
            st.subheader("📊 Live Sentiment Results")
            st.dataframe(results_df, use_container_width=True)
    else:
        st.error("No news articles available.")

else:  # About Project
    st.title("📚 About the Project")
    
    st.markdown("""
    ## Sentiment Classification of Financial News Using NLP
    
    ### 🎯 Project Overview
    This project implements an advanced sentiment analysis system specifically designed for 
    financial text. Using state-of-the-art Natural Language Processing techniques, we classify 
    financial news, reports, and commentary into positive, negative, or neutral sentiments.
    
    ### 🔬 Methodology
    
    #### 1. Dataset
    - **Source:** Financial PhraseBank
    - **Size:** 4,840+ expert-annotated sentences
    - **Classes:** Positive, Negative, Neutral
    
    #### 2. Approach
    We implemented and compared multiple approaches:
    
    **Traditional ML Models (with TF-IDF):**
    - Logistic Regression
    - Naive Bayes
    - Support Vector Machine (SVM)
    - Random Forest
    
    **Transformer Model:**
    - FinBERT (BERT trained on processed financial text data)
    
    #### 3. Preprocessing Pipeline
    - Text cleaning and normalization
    - Tokenization
    - Stopword removal (with financial term preservation)
    - Lemmatization
    
    ### 📊 Results
    
    Our transformer-based model (FinBERT) achieved:
    - **Validation Accuracy:** ~89.8%
    - **Testing Accuracy:** ~87.6%
    - **F1-Score:** ~0.89
    - Superior performance on domain-specific financial terminology
    
    ### 🛠️ Technical Stack
    - **Framework:** Streamlit
    - **ML Libraries:** Transformers, PyTorch, Scikit-learn
    - **Visualization:** Plotly, Matplotlib
    - **NLP:** NLTK, Hugging Face
    
    ### 💡 Applications
    - **Investment Analysis for Traders:** Support trading decisions
    - **Risk Management:** Early detection of negative trends
    - **Market Research:** Automated sentiment monitoring
    - **Portfolio Management:** Sentiment-based asset allocation
    
    ### 🎓 Academic Context
    **Course:** CT-485 - Natural Language Processing  
    **Institution:** NED University of Engineering & Technology  
    **Department:** Computer Science and Information Technology
    
    ### 👥 Team Members
    - **Laiba Masood** (CT-22001)
    - **Aiza Asim** (CT-22006)
    - **Zainab Fatima** (CT-22007)
    - **Maryam Shaikh** (CT-22012)
    
    ### 📚 References
    1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models
    2. Malo, P., et al. (2014). Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts
    3. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
    
    ### 🔗 Resources
    - [Financial PhraseBank Dataset](https://huggingface.co/datasets/takala/financial_phrasebank)
    - [FinBERT Model](https://huggingface.co/ProsusAI/finbert)
    - [Project Repository](#)
    
    ---
    
    *Last Updated: November 2025*
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Financial Sentiment Analyzer v1.0 | NED University | CT-485 NLP Project 2025</p>
</div>
""", unsafe_allow_html=True)
