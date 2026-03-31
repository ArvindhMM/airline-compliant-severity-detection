import streamlit as st

# Page config
st.set_page_config(
    page_title="Airline Complaint Analyzer",
    page_icon="✈️",
    layout="centered"
)

# Title
st.title("✈️ Airline Complaint Severity & Risk Analyzer")

# Subtitle
st.markdown("""
Welcome! This app analyzes airline complaints using Machine Learning & Deep Learning models.

### 🔍 What this app does:
- Classifies complaint **severity**:
  - 🔴 Critical
  - 🟠 High
  - 🟡 Medium
  - 🟢 Low
- Detects **high-risk complaints** ⚠️

---

### 🤖 Models Used:
- Logistic Regression (TF-IDF)
- Naive Bayes
- SVM
- CNN & LSTM
- DistilBERT (Transformer - Best Model)

---

### 🚀 How to use:
1. Go to **Severity Prediction** page
2. Enter complaint text
3. View predictions instantly

---

### 💡 Example Input:
> "My flight was delayed for 6 hours and customer service was terrible"

---

Built as part of an AI project 🚀
""")

# Footer
st.markdown("---")
st.caption("Developed by Arvindh | AI Airline Complaint Detection Project")