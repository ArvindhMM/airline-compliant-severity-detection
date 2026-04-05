import streamlit as st
import pickle

st.set_page_config(layout="wide")

# -----------------------------
# Load Model + Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("models/risk_model.pkl", "rb"))
    vectorizer = pickle.load(open("models/tfidf_risk.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("⚠️ Risk Prediction (ML Model)")

user_input = st.text_area("Enter airline complaint:")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Risk"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:
        # 🔥 Spinner for better UX
        with st.spinner("Analyzing complaint..."):

            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0]

        # -----------------------------
        # Prediction Result
        # -----------------------------
        st.markdown("---")

        if prediction == 1:
            st.error("🚨 High Risk Complaint")
        else:
            st.success("✅ Low Risk Complaint")

        st.write("Confidence:", round(max(prob), 4))

        # -----------------------------
        # Explainability
        # -----------------------------
        st.markdown("---")
        st.subheader("🔍 Important Words Influencing Prediction")

        try:
            feature_names = vectorizer.get_feature_names_out()
            importance = model.coef_[0]

            # High risk words
            top_positive = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            # Low risk words
            top_negative = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1]
            )[:5]

            st.write("🔴 High Risk Indicators:")
            st.write([w[0] for w in top_positive])

            st.write("🟢 Low Risk Indicators:")
            st.write([w[0] for w in top_negative])

        except:
            st.warning("Explainability not available for this model")

# -----------------------------
# CSV BULK PREDICTION
# -----------------------------
st.markdown("---")
st.subheader("📂 Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df_upload.head())

    # 🔍 Let user select text column
    text_column = st.selectbox(
        "Select column containing complaint text",
        df_upload.columns
    )

    if st.button("Run Bulk Prediction"):

        with st.spinner("Processing file..."):

            try:
                X = vectorizer.transform(df_upload[text_column].astype(str))
                preds = model.predict(X)
                probs = model.predict_proba(X)

                df_upload["prediction"] = preds
                df_upload["confidence"] = probs.max(axis=1)

                # Convert labels (optional)
                df_upload["prediction_label"] = df_upload["prediction"].map({
                    1: "High Risk",
                    0: "Low Risk"
                })

                st.success("✅ Predictions completed!")

                st.dataframe(df_upload)

                # 📥 Download button
                csv = df_upload.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Results",
                    csv,
                    "risk_predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Error processing file: {e}")