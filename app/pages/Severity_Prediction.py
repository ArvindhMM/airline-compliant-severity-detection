import streamlit as st
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

st.title("✈️ Severity Prediction (BERT Model)")

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    model_path = os.path.join(BASE_DIR, "models", "bert_model")

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

# -------------------------------
# Label mapping (IMPORTANT)
# -------------------------------
labels_map = {
    0: "Critical 🔴",
    1: "High 🟠",
    2: "Medium 🟡",
    3: "Low 🟢"
}

# -------------------------------
# Input
# -------------------------------
user_input = st.text_area("Enter airline complaint text:")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Severity"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing..."):

            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                probs = F.softmax(logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)

                pred_class = pred.item()
                confidence_score = confidence.item()

        # -------------------------------
        # Display results
        # -------------------------------
        st.subheader("Prediction Result")

        st.write(f"**Severity:** {labels_map[pred_class]}")
        st.write(f"**Confidence:** {confidence_score:.2f}")

        # Optional: show all class probabilities
        st.subheader("Class Probabilities")
        for i, prob in enumerate(probs[0]):
            st.write(f"{labels_map[i]}: {prob.item():.2f}")