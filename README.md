# 📚 Sentiment Classification of Financial News Using NLP

## 🎯 Project Overview

This project implements an advanced sentiment analysis system tailored
for financial text. Using modern Natural Language Processing (NLP)
techniques, the system classifies financial news, reports, and
commentary into **positive**, **negative**, or **neutral** sentiment
categories.

## 🔬 Methodology

### 1. Dataset

-   **Source:** Financial PhraseBank\
-   **Size:** 4,840+ expert-annotated sentences\
-   **Classes:** Positive, Negative, Neutral

### 2. Approach

We implemented and compared two major modeling strategies:

#### Traditional ML Models (TF-IDF based):

-   Logistic Regression\
-   Naive Bayes\
-   Support Vector Machine (SVM)\
-   Random Forest

#### Transformer-based Model:

-   **FinBERT** --- A BERT variant fine-tuned on large-scale financial
    text data

### 3. Preprocessing Pipeline

-   Text cleaning and normalization\
-   Tokenization\
-   Stopword removal (financial terms preserved)\
-   Lemmatization

## 📊 Results

Our transformer-based model **FinBERT** delivered the strongest
performance:

  Metric                    Score
  ------------------------- ---------
  **Validation Accuracy**   \~89.8%
  **Testing Accuracy**      \~87.6%
  **F1-Score**              \~0.89

FinBERT also demonstrated superior handling of specialized financial
terminology.

## 🛠️ Technical Stack

-   **Frontend Framework:** Streamlit\
-   **ML Libraries:** Transformers, PyTorch, Scikit-learn\
-   **Visualization:** Plotly, Matplotlib\
-   **NLP Tools:** NLTK, Hugging Face

## 💡 Applications

-   **Investment Analysis:** Enhance trading decisions\
-   **Risk Management:** Detect negative trends early\
-   **Market Research:** Automated sentiment monitoring\
-   **Portfolio Management:** Sentiment-driven asset strategies

## 🎓 Academic Context

-   Course: CT-485 -- Natural Language Processing\
-   Institution: NED University of Engineering & Technology\
-   Department: Computer Science & Information Technology

## 📚 References

-   Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with
    Pre-trained Language Models*\
-   Malo, P., et al. (2014). *Good Debt or Bad Debt: Detecting Semantic
    Orientations in Economic Texts*\
-   Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional
    Transformers*

## 🔗 Resources

-   Financial PhraseBank Dataset\
-   FinBERT Model\
-   Project Repository

------------------------------------------------------------------------

### ❤️ *Made with love for finance, stock market, traders and investors.*
