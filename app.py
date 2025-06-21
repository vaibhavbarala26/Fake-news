from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
model_path = "Vaibhavbarala/Fake-news"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Model and tokenizer loaded successfully.")
print(f"Model: {model.eval()}")

def predict_label(text):
    import re  # Ensure re is imported in case it's not already

    # Clean the input (same as training)
    def clean(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    text = clean(text)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).numpy()[0]  # Get first example

    predicted_class = np.argmax(probs)
    label_map = {0: "Fake", 1: "Real"}
    predicted_label = label_map[predicted_class]

    # Convert to percentage format
    fake_pct = probs[0] * 100
    real_pct = probs[1] * 100

    return {
        "label": predicted_label,
        "confidence": {
            "Fake": f"{fake_pct:.2f}%",
            "Real": f"{real_pct:.2f}%"
        }
    }

