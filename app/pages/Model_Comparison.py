import streamlit as st
import pandas as pd

st.title("📊 Model Comparison")

st.markdown("Compare performance of different models used in this project.")

# Load results
@st.cache_data
def load_results():
    return pd.read_csv("data/model_results.csv")

try:
    df = load_results()

    st.subheader("📋 Model Performance Table")
    st.dataframe(df)

    # Highlight best model
    best_model = df.sort_values(by="F1 Score", ascending=False).iloc[0]

    st.success(f"🏆 Best Model: {best_model['Model']}")

    # Bar chart
    st.subheader("📈 Model Comparison (F1 Score)")
    st.bar_chart(df.set_index("Model")["F1 Score"])

except Exception as e:
    st.error(f"Error loading results: {e}")