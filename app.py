import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Student Performance Intelligence",
    page_icon="🎓",
    layout="wide"
)

# Load project artifacts
model = joblib.load(os.path.join(BASE_DIR, "student_risk_model.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
training_features = pd.read_csv(
    os.path.join(BASE_DIR, "student_features.csv")
)
comparison = pd.read_csv(
    os.path.join(BASE_DIR, "model_comparison.csv")
)
cm = pd.read_csv(
    os.path.join(BASE_DIR, "confusion_matrix.csv"),
    index_col=0
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #101b2d);
    color: #edf4ff;
}
[data-testid="stMetric"] {
    background: #162438;
    padding: 18px;
    border: 1px solid #29435b;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #e7f5ff;
}
div.stButton > button {
    background: #13b8a6;
    color: #071522;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎓 Student Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict Student Risk", "Model Performance"]
)

st.sidebar.caption("Academic risk screening prototype")
st.sidebar.caption("Dataset: UCI Student Performance")

# Header
st.title("Student Performance Prediction")
st.write(
    "An interactive machine learning system for identifying "
    "students who may need additional academic support."
)

if page == "Overview":
    total = len(training_features)
    at_risk = int((pd.read_csv(
        os.path.join(BASE_DIR, "student_features.csv")
    ).shape[0] * 0.3291))

    best = comparison.sort_values(
        "F1-Score", ascending=False
    ).iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", total)
    c2.metric("Input Features", len(feature_columns))
    c3.metric("Best Model", best["Model"])

    st.subheader("Model Comparison")
    st.dataframe(
        comparison.sort_values("F1-Score", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "At-risk means a final Mathematics grade below 10/20 "
        "in the historical dataset. This prototype is intended "
        "for screening and educational support, not high-stakes decisions."
    )

elif page == "Predict Student Risk":
    st.subheader("Student Assessment")
    st.write(
        "Enter the student's available background and school-related "
        "information. Final grades are not used as model inputs."
    )

    numeric_columns = training_features.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        col for col in training_features.columns
        if col not in numeric_columns
    ]

    with st.form("student_form"):
        st.markdown("#### Student Information")
        inputs = {}
        left, right = st.columns(2)

        for i, col in enumerate(training_features.columns):
            container = left if i % 2 == 0 else right

            with container:
                if col in categorical_columns:
                    options = sorted(
                        training_features[col].dropna().unique().tolist()
                    )
                    inputs[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options=options
                    )
                else:
                    min_value = int(training_features[col].min())
                    max_value = int(training_features[col].max())
                    default_value = int(
                        training_features[col].median()
                    )
                    inputs[col] = st.number_input(
                        col.replace("_", " ").title(),
                        min_value=min_value,
                        max_value=max_value,
                        value=default_value,
                        step=1
                    )

        submitted = st.form_submit_button(
            "Analyze Student Risk",
            use_container_width=True
        )

    if submitted:
        new_student = pd.DataFrame([inputs])

        # Encode using the original training data so that
        # the one-hot encoding baseline remains consistent.
        combined = pd.concat(
            [training_features, new_student],
            ignore_index=True
        )
        encoded = pd.get_dummies(
            combined,
            drop_first=True,
            dtype=int
        )

        model_input = encoded.iloc[[-1]].reindex(
            columns=feature_columns,
            fill_value=0
        )

        prediction = int(model.predict(model_input)[0])
        probabilities = model.predict_proba(model_input)[0]
        risk_probability = float(probabilities[1])

        st.divider()
        st.subheader("Prediction Results")

        a, b = st.columns(2)
        a.metric(
            "Estimated At-Risk Probability",
            f"{risk_probability:.1%}"
        )
        b.metric(
            "Predicted Category",
            "At Risk" if prediction == 1 else "Not At Risk"
        )

        st.progress(risk_probability)

        if prediction == 1:
            st.warning(
                "The model flags this student for additional review. "
                "Consider checking attendance, study support, and "
                "individual learning needs."
            )
        else:
            st.success(
                "The model does not flag this student as at risk. "
                "Continue monitoring academic progress as usual."
            )

        st.caption(
            "This is a model estimate, not a diagnosis or a guarantee "
            "of future academic performance. Review predictions with "
            "a teacher and the student's actual circumstances."
        )

elif page == "Model Performance":
    st.subheader("Evaluation on the Held-Out Test Set")
    st.write(
        "All reported metrics come from the saved 20% test split."
    )

    st.dataframe(
        comparison.sort_values("F1-Score", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Compare models by",
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    )

    chart_data = comparison.set_index("Model")[[metric]]
    st.bar_chart(chart_data)

    st.subheader("Confusion Matrix")
    st.dataframe(cm, use_container_width=True)

    st.subheader("Classification Report")
    report_path = os.path.join(
        BASE_DIR, "classification_report.txt"
    )
    with open(report_path, "r") as file:
        st.code(file.read(), language="text")
