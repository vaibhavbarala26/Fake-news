import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import re

# -------- Page Setup --------
st.set_page_config(
    page_title="Fake News Detector", 
    page_icon="📰", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------- Custom CSS --------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #ffffff;
    color: #1f2937;
}

/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1rem;
}

/* Force text colors */
.stApp, .stApp * {
    color: #1f2937 !important;
}

/* Specific text elements */
p, div, span, label {
    color: #1f2937 !important;
}

/* Header styling */
.header-section {
    text-align: center;
    margin-bottom: 3rem;
    padding: 2rem 0;
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 1rem;
    letter-spacing: -0.025em;
}

.subtitle {
    font-size: 1.25rem;
    color: #6b7280;
    font-weight: 400;
    line-height: 1.6;
    max-width: 600px;
    margin: 0 auto;
}

/* Input section */
.input-section {
    background: transparent;
    border-radius: 0px;
    padding: 0rem;
    margin-bottom: 0rem;
    border: none;
}

.stTextArea > div > div > textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    border-radius: 12px !important;
    border: 2px solid #e5e7eb !important;
    background-color: #ffffff !important;
    color: #000000 !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
    resize: vertical !important;
}

.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    outline: none !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    width: 100% !important;
    margin-top: 1rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4) !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Result card */
.result-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 2.5rem;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    border: 1px solid #f3f4f6;
    text-align: center;
}

.prediction-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.fake-prediction {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    color: #dc2626;
    border: 2px solid #fca5a5;
}

.real-prediction {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    color: #059669;
    border: 2px solid #6ee7b7;
}

/* Percentage display */
.percentage-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 2rem;
}

.percentage-card {
    background: #f8fafc;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.percentage-card.fake {
    border-color: #fee2e2;
}

.percentage-card.real {
    border-color: #d1fae5;
}

.percentage-label {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}

.percentage-label.fake {
    color: #dc2626;
}

.percentage-label.real {
    color: #059669;
}

.percentage-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}

.percentage-value.fake {
    color: #b91c1c;
}

.percentage-value.real {
    color: #047857;
}

/* Expandable sections */
.stExpander {
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    margin-bottom: 1rem !important;
    background: #ffffff !important;
}

.stExpander > div {
    background: #ffffff !important;
    color: #1f2937 !important;
}

.stExpander > div > div {
    background: #ffffff !important;
    color: #1f2937 !important;
}

.stExpander > div > div > div {
    background: #ffffff !important;
    color: #1f2937 !important;
}

.stExpander > div > div > div > div {
    background: #ffffff !important;
    color: #1f2937 !important;
}

.stExpander summary {
    color: #1f2937 !important;
}

.stExpander p, .stExpander div, .stExpander span {
    color: #1f2937 !important;
}

/* Warning styling */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    padding: 1rem 1.5rem !important;
    color: #1f2937 !important;
}

.stAlert p, .stAlert div, .stAlert span {
    color: #1f2937 !important;
}

/* General text color enforcement */
.stMarkdown, .stMarkdown p, .stMarkdown div {
    color: #1f2937 !important;
}

/* Spinner text */
.stSpinner > div {
    color: #1f2937 !important;
}

/* Input label */
.stTextArea > label {
    color: #1f2937 !important;
}

/* Footer */
.custom-footer {
    text-align: center;
    margin-top: 4rem;
    padding: 2rem 0;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
    font-size: 0.9rem;
}

.custom-footer a {
    color: #3b82f6;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
}

.custom-footer a:hover {
    color: #1d4ed8;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# -------- Load Model & Tokenizer --------
@st.cache_resource
def load_model():
    model_name = "Vaibhavbarala/Fake-news"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# -------- Utility Functions --------
def clean_text(text):
    """Clean and preprocess text for model input"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_news(text):
    """Predict if news is fake or real"""
    cleaned_text = clean_text(text)
    inputs = tokenizer(
        cleaned_text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    return np.argmax(probabilities), probabilities

# -------- Header Section --------
st.markdown("""
<div class="header-section">
    <h1 class="main-title">📰 Fake News Detector</h1>
    <p class="subtitle">
        Detect whether a news article or headline is <strong>Fake</strong> or <strong>Real</strong> 
        using advanced AI powered by BERT and trained on the WELFake dataset.
    </p>
</div>
""", unsafe_allow_html=True)

# -------- Input Section --------
text_input = st.text_area(
    "✏️ Enter a news headline or article:",
    height=150,
    placeholder="Example: Breaking news: Scientists discover that eating chocolate daily increases intelligence by 200%...",
    help="Paste any news headline or article text you want to verify"
)

analyze_button = st.button("🔍 Analyze News", use_container_width=True)

# -------- Analysis Section --------
if analyze_button:
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🤖 Analyzing with AI..."):
            try:
                label_idx, probabilities = predict_news(text_input)
                labels = ["Fake", "Real"]
                predicted_label = labels[label_idx]
                confidence = probabilities[label_idx] * 100
                
                fake_percentage = probabilities[0] * 100
                real_percentage = probabilities[1] * 100
                
                # Result card
              
                # Prediction badge
                if predicted_label == "Fake":
                    st.markdown(f"""
                    <div class="prediction-badge fake-prediction">
                        ❌ Likely Fake News
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-badge real-prediction">
                        ✅ Likely Real News
                    </div>
                    """, unsafe_allow_html=True)
                
                # Percentage display
                st.markdown("""
                <div class="percentage-container">
                    <div class="percentage-card fake">
                        <div class="percentage-label fake">Fake News</div>
                        <div class="percentage-value fake">{:.1f}%</div>
                    </div>
                    <div class="percentage-card real">
                        <div class="percentage-label real">Real News</div>
                        <div class="percentage-value real">{:.1f}%</div>
                    </div>
                </div>
                """.format(fake_percentage, real_percentage), unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional info
                if confidence < 70:
                    st.info("💡 **Note:** The confidence is moderate. Consider verifying with additional sources.")
                elif confidence > 90:
                    st.success("🎯 **High Confidence:** The model is very confident in this prediction.")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")

# -------- Information Sections --------
with st.expander("🔬 How It Works"):
    st.markdown("""
    This application uses a sophisticated **BERT** (Bidirectional Encoder Representations from Transformers) model 
    that has been fine-tuned specifically for fake news detection.
    
    **Process:**
    1. **Text Preprocessing**: Your input is cleaned and tokenized
    2. **AI Analysis**: The BERT model analyzes linguistic patterns, context, and semantic meaning
    3. **Classification**: Output probabilities for both 'Fake' and 'Real' categories
    4. **Results**: Clear visualization of the prediction with confidence scores
    """)

with st.expander("📊 Model & Dataset Information"):
    st.markdown("""
    **Model Details:**
    - **Base Model**: `bert-base-uncased` (110M parameters)
    - **Fine-tuning**: Specialized for fake news detection
    - **Architecture**: Transformer-based neural network
    
    **Dataset:**
    - **Source**: [WELFake Dataset](https://aclanthology.org/2020.fever-1.9/)
    - **Content**: Mix of real news articles and AI-generated fake news
    - **Size**: Thousands of labeled examples for training
    
    **Technology Stack:**
    - Hugging Face Transformers
    - PyTorch
    - Streamlit
    """)

with st.expander("📈 Understanding Confidence Scores"):
    st.markdown("""
    **Interpreting Results:**
    
    - **90-100%**: Very high confidence - Strong indicators present
    - **70-89%**: High confidence - Clear patterns detected  
    - **60-69%**: Moderate confidence - Some uncertainty remains
    - **50-59%**: Low confidence - Consider additional verification
    
    **Important Notes:**
    - No AI model is 100% accurate
    - Always cross-reference with reliable news sources
    - Be especially cautious with moderate confidence scores
    - Context and source credibility matter significantly
    """)

# -------- Footer --------
st.markdown("""
<div class="custom-footer">
    <p>
        Made with ❤️ by <strong>Vaibhav Barala</strong> • 
        <a href="https://github.com/Vaibhavbarala26" target="_blank">GitHub</a> • 
        <a href="https://medium.com/@Vaibhavbarala8" target="_blank">Medium</a>
    </p>
    <p style="margin-top: 0.5rem; font-size: 0.8rem;">
        Always verify news from multiple reliable sources. This tool is for educational purposes.
    </p>
</div>
""", unsafe_allow_html=True)