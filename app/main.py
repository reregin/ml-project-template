"""Optional Streamlit client for the domain-neutral Phase 1 example."""

import streamlit as st

from ml_project_template.config import get_settings
from ml_project_template.predict import predict_record

settings = get_settings()

st.set_page_config(page_title="ML Project Template")
st.title("Generic Binary Classification Example")
st.caption("This form exists to prove the raw-input-to-model contract works.")

with st.form("prediction_form"):
    numeric_feature_1 = st.number_input("numeric_feature_1", value=0.25)
    numeric_feature_2 = st.number_input("numeric_feature_2", value=-0.50)
    category_feature_1 = st.selectbox(
        "category_feature_1",
        ["category_a", "category_b", "category_c", "unseen_category"],
    )
    category_feature_2 = st.selectbox(
        "category_feature_2",
        ["group_1", "group_2"],
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    record = {
        "numeric_feature_1": numeric_feature_1,
        "numeric_feature_2": numeric_feature_2,
        "category_feature_1": category_feature_1,
        "category_feature_2": category_feature_2,
    }
    try:
        result = predict_record(record, settings.model_artifact_path)
        st.success(f"Prediction: {result['prediction']}")
        if result["probability"] is not None:
            st.metric("Class 1 probability", f"{result['probability']:.2%}")
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
