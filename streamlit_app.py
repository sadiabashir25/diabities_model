import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="centered"
)

# --- CSS for Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .low-risk { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .mod-risk { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .high-risk { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_assets():
    model = joblib.load('diabetes_ensemble_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    ensemble_model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- Header ---
st.title("🏥 Diabetes Risk Prediction Tool")
st.markdown("""
Enter the patient's diagnostic measurements below to assess the risk of diabetes onset. 
This tool uses an Ensemble Machine Learning model trained on the Pima Indians Dataset.
""")

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, help="Number of times pregnant")
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, help="Plasma glucose concentration")
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, help="Diastolic blood pressure (mm Hg)")
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness (mm)")

    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=80, help="2-Hour serum insulin (mu U/ml)")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f", help="Body mass index")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f", help="Genetic diabetes risk factor")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Age in years")

    submit_button = st.form_submit_button("Predict Diabetes Risk")

# --- Prediction Logic ---
if submit_button:
    # 1. Create feature array
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                         insulin, bmi, dpf, age]])

    # 2. Handle missing values (0s) - match original project logic
    # Median values used in the training script for replacement
    median_values = [117.0, 72.0, 29.0, 125.0, 32.3] # Glucose, BP, Skin, Insulin, BMI
    for idx, median_val in enumerate(median_values):
        if features[0][idx+1] == 0:
            features[0][idx+1] = median_val

    # 3. Scale features
    features_scaled = scaler.transform(features)

    # 4. Predict
    prediction = ensemble_model.predict(features_scaled)[0]
    probability = ensemble_model.predict_proba(features_scaled)[0][1]

    # --- Display Results ---
    st.subheader("Results")
    
    prob_percent = round(probability * 100, 2)
    
    if probability < 0.3:
        risk_class = "low-risk"
        risk_level = "Low"
        emoji = "✅"
        msg = "The patient is at low risk. Maintain a healthy lifestyle."
    elif probability < 0.7:
        risk_class = "mod-risk"
        risk_level = "Moderate"
        emoji = "⚠️"
        msg = "The patient is at moderate risk. Consultation with a provider is recommended."
    else:
        risk_class = "high-risk"
        risk_level = "High"
        emoji = "🚨"
        msg = "The patient is at high risk. Please consult a healthcare professional immediately."

    st.markdown(f"""
    <div class="result-box {risk_class}">
        <h2>{emoji} Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}</h2>
        <h3>Probability: {prob_percent}%</h3>
        <h4>Risk Level: {risk_level}</h4>
        <p>{msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # Optional: Expandable details
    with st.expander("View Input Data Summary"):
        input_df = pd.DataFrame(features, columns=['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age'])
        st.table(input_df)

# --- Footer ---
st.divider()
st.caption("Disclaimer: This tool is for educational and screening purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.")
