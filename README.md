# Airline Complaint Analysis & Risk Classification System

## Objective
This project analyzes airline customer complaints using Machine Learning and Deep Learning techniques to:

- Classify complaint **severity** (Low, Medium, High, Critical)
- Detect **high-risk complaints** requiring urgent attention
- Provide **insights** through an EDA dashboard
- Enable **real-time and bulk predictions** via a Streamlit application

## Dataset
**Airline customer review dataset** containing:
- Airline Name
- Overall Rating
- Review Text
- Verified Status
- Recommendation

## 🔹 Preprocessing
- Cleaned and normalized text data
- Converted ratings to numeric
- Created `severity_label`:
  
| Rating | Severity  |
|--------|-----------|
| 1–2    | Critical  |
| 3–4    | High      |
| 5–6    | Medium    |
| 7+     | Low       |

## Models Used

### 🔹 Traditional ML Models (TF-IDF)
- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

**Used for**: Risk Prediction (High / Low)

### 🔹 Deep Learning Model
- **DistilBERT** (Transformer - HuggingFace)

**Used for**: Severity Classification

## Streamlit Application Features

### 1. Home Page
- Project overview
- Model explanation
- Features and business use cases

### 2. EDA Dashboard (Interactive)
- Severity Distribution (**Pie Chart**)
- Airline-wise Complaints (**Bar Chart**)
- Complaint Length Distribution (**Histogram**)
- Top Words (**NLP insights**)
- Severity by Airline (**Stacked Bar**)
- Average Rating by Airline
- Verified vs Non-Verified Reviews
- Recommendation Distribution (**Donut Chart**)
- Complaint Length vs Severity

**Built using Plotly** for interactivity

### 3. Model Comparison
Compared multiple models using:
- Accuracy
- Precision
- Recall
- **F1 Score**

**Identified best-performing model**

### 4. Risk Prediction (ML Model)
- **Single Prediction**: Input complaint text → High Risk / Low Risk + Confidence score
- **Explainability**: Shows important words influencing prediction
- **Bulk Prediction**: CSV Upload → Select text column → Predict all rows → Download results

### 5. Severity Prediction (BERT)
- Input: Complaint text
- Output: Severity class + Confidence score + Class probabilities

## UI/UX Features
- Multi-page Streamlit app
- Dashboard-style layout
- Interactive charts (Plotly)
- Loading indicators
- Clean navigation
- Downloadable outputs

## Key Features
- Real-time prediction
- Bulk CSV processing
- Explainable ML outputs
- Interactive EDA
- Hybrid ML + Deep Learning system

## Business Use Cases
- Prioritize **critical customer complaints**
- Improve **airline customer support**
- Detect **negative sentiment early**
- Enhance **customer satisfaction**
- Enable **data-driven decisions**

## 🛠️ Tech Stack
- Python | Pandas | NumPy | Scikit-learn | HuggingFace Transformers | Streamlit | Plotly


## Project Structure
project/
│
├── models/
│ ├── risk_model.pkl
│ ├── tfidf_risk.pkl
│ └── bert_model/ # Not included (large size)
│
├── data/
│ └── clean_airline_reviews.csv
│
├── app/
│ ├── Home.py
│ └── pages/
│ ├── EDA_Dashboard.py
│ ├── Model_Comparison.py
│ ├── Risk_Prediction.py
│ └── Severity_Prediction.py
│
├── notebooks/
├── requirements.txt
└── README.md

## How to Run
```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

## Note
**DistilBERT model** is not included due to size limitations.

**To recreate it, run:**
```bash
notebooks/transformer_models.ipynb
```

##  GitHub Repository
You can view the complete source code here:

[View on GitHub](https://github.com/ArvindhMM/airline-compliant-severity-detection)

