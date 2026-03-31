import streamlit as st
import pickle

# Load model + vectorizer
@st.cache_resource
def load_model():
    model = pickle.load(open("models/risk_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/tfidf_risk.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# UI
st.title("⚠️ Risk Prediction (ML Model)")

user_input = st.text_area("Enter airline complaint:")

if st.button("Predict Risk"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        if prediction == 1:
            st.error("🚨 High Risk Complaint")
        else:
            st.success("✅ Low Risk Complaint")

        st.write("Confidence:", max(prob))