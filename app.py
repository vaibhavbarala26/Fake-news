import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import re

# -------- Page Setup --------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# -------- Custom CSS --------
st.markdown("""
<style>
body {
    background-color: #f9fafb;
    color: #1f2937;
}
.stApp {
    max-width: 750px;
    margin: auto;
    font-family: "Segoe UI", sans-serif;
}
h1, h2, h3 {
    color: #111827;
}
.description {
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
    color: #4b5563;
}
.stTextArea textarea {
    font-size: 16px;
    padding: 0.75rem;
    border-radius: 10px;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    font-weight: bold;
    font-size: 16px;
    padding: 0.6em 1.5em;
    border-radius: 10px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #1e40af;
}
.result-box {
    padding: 1.5rem;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    text-align: center;
    margin-top: 1rem;
}
.badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1rem;
    margin-top: 0.5rem;
}
.fake-badge {
    background-color: #fee2e2;
    color: #b91c1c;
}
.real-badge {
    background-color: #d1fae5;
    color: #065f46;
}
footer {
    text-align: center;
    margin-top: 3rem;
    color: #6b7280;
    font-size: 0.9rem;
}
a {
    color: #2563eb;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# -------- Load Model & Tokenizer --------
model_name = "Vaibhavbarala/Fake-news"  # Change if needed
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# -------- Utility Functions --------
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict(text):
    text = clean(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]  # ✅ Fix: move to CPU before numpy
    return np.argmax(probs), probs


# -------- UI Content --------
st.title("📰 Fake News Detector")
st.markdown("<p class='description'>Quickly detect whether a news article or headline is <b>Fake</b> or <b>Real</b> using a BERT-based model fine-tuned on news data.</p>", unsafe_allow_html=True)

text_input = st.text_area("✏️ Enter a news headline or article:", height=150, placeholder="e.g., Breaking: Scientists discover talking dogs in Antarctica...")

if st.button("🔍 Analyze"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        label_idx, probs = predict(text_input)
        labels = ["Fake", "Real"]
        predicted_label = labels[label_idx]
        confidence = probs[label_idx] * 100

        if predicted_label == "Fake":
               st.markdown(f"""
                     <div class='result-box'>
                       <div class='badge fake-badge'>❌ Fake News – Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
                st.markdown(f"""
                     <div class='result-box'>
                       <div class='badge real-badge'>✅ Real News – Confidence: {confidence:.2f}%</div>
                    </div>
                   """, unsafe_allow_html=True)
       
        

# -------- Expandable Sections --------
with st.expander("📊 How It Works"):
    st.write("""
    This app uses a pre-trained BERT (`bert-base-uncased`) model fine-tuned on the WELFake dataset for binary classification.
    The input text is tokenized and passed through the model, which outputs probabilities for the 'Fake' and 'Real' classes.
    """)

with st.expander("🧠 Model & Dataset Details"):
    st.markdown("""
    - **Model:** `bert-base-uncased` fine-tuned for fake news detection  
    - **Dataset:** [WELFake](https://aclanthology.org/2020.fever-1.9/) (real + generated news articles)  
    - **Frameworks:** Hugging Face Transformers, PyTorch, Streamlit  
    """)

with st.expander("📁 Confidence Score Explanation"):
    st.markdown("""
    The confidence score reflects the model’s certainty.  
    - A score close to 100% indicates high certainty  
    - 50%-70% may warrant further human review
    """)

# -------- Footer --------
st.markdown("""
<footer>
    Made with ❤️ by <b>Vaibhav Barala</b> •  
    <a href="https://github.com/Vaibhavbarala26" target="_blank">GitHub</a> •  
    <a href="https://medium.com/@Vaibhavbarala8" target="_blank">Medium</a>
</footer>
""", unsafe_allow_html=True)
