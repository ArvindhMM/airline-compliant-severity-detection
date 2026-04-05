import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(layout="wide")
st.title("📊 EDA Dashboard")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/clean_airline_reviews.csv")
    
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    return df

df = load_data()

# -----------------------------
# Create Severity Label
# -----------------------------
def severity_label(rating):
    if rating <= 2:
        return "Critical"
    elif rating <= 4:
        return "High"
    elif rating <= 6:
        return "Medium"
    else:
        return "Low"

df["overall_rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")
df["severity_label"] = df["overall_rating"].apply(severity_label)

# Create text length
df["text_length"] = df["review"].astype(str).apply(len)

# -----------------------------
# TOP METRICS
# -----------------------------
col1, col2 = st.columns(2)

col1.metric("Total Samples", len(df))
col2.metric("Classes", df["severity_label"].nunique())

st.markdown("---")

# -----------------------------
# ROW 1: Severity + Airline Complaints
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Severity Distribution")
    severity_counts = df["severity_label"].value_counts()
    fig = px.pie(
        names=severity_counts.index,
        values=severity_counts.values,
        title="Severity Breakdown"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("✈️ Airline Complaints")
    airline_counts = df["airline_name"].value_counts().head(10)
    fig = px.bar(
        x=airline_counts.values,
        y=airline_counts.index,
        orientation='h',
        title="Top Airlines by Complaints"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# ROW 2: Length + Top Words
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Complaint Length")
    fig = px.histogram(df, x="text_length", nbins=30, title="Text Length Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔤 Top Words")
    all_text = " ".join(df["review"].astype(str))
    words = Counter(all_text.split()).most_common(10)
    words_df = pd.DataFrame(words, columns=["Word", "Count"])

    fig = px.bar(words_df, x="Count", y="Word", orientation='h', title="Top Words")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# ROW 3: Severity by Airline + Avg Rating
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("✈️ Severity by Airline")
    top_airlines = df["airline_name"].value_counts().head(5).index
    filtered_df = df[df["airline_name"].isin(top_airlines)]

    pivot = pd.crosstab(
        filtered_df["airline_name"],
        filtered_df["severity_label"]
    ).reset_index()

    fig = px.bar(
        pivot,
        x="airline_name",
        y=["Critical", "High", "Medium", "Low"],
        title="Severity Distribution per Airline"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⭐ Avg Rating by Airline")
    avg_rating = (
        df.groupby("airline_name")["overall_rating"]
        .mean()
        .sort_values()
        .head(10)
    )

    fig = px.bar(
        x=avg_rating.values,
        y=avg_rating.index,
        orientation='h',
        title="Average Ratings"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# ROW 4: Verified + Recommendation
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Verified Reviews")
    verified_counts = df["verified"].value_counts()

    fig = px.pie(
        names=verified_counts.index,
        values=verified_counts.values,
        title="Verified vs Non-Verified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("👍 Recommendation")
    rec_counts = df["recommended"].value_counts()

    fig = px.pie(
        names=rec_counts.index,
        values=rec_counts.values,
        hole=0.4,
        title="Recommendation Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# ROW 5: Length vs Severity
# -----------------------------
st.subheader("📝 Length vs Severity")

length_severity = (
    df.groupby("severity_label")["text_length"]
    .mean()
    .reset_index()
)

fig = px.bar(
    length_severity,
    x="severity_label",
    y="text_length",
    title="Avg Complaint Length by Severity"
)

st.plotly_chart(fig, use_container_width=True)