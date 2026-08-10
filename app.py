import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AP Mechanical Ventilation Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical UX
st.markdown(
    """
    <style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
        padding: 12px;
        background-color: #EFF6FF;
        border-radius: 8px;
        border-left: 5px solid #2563EB;
    }
    .sub-header {
        font-size: 16px;
        font-weight: bold;
        color: #1E293B;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .result-box-high {
        padding: 20px;
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        border-radius: 8px;
        margin-top: 15px;
    }
    .result-box-mid {
        padding: 20px;
        background-color: #FFFBEB;
        border-left: 6px solid #D97706;
        border-radius: 8px;
        margin-top: 15px;
    }
    .result-box-low {
        padding: 20px;
        background-color: #F0FDF4;
        border-left: 6px solid #16A34A;
        border-radius: 8px;
        margin-top: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">Mechanical Ventilation Risk Prediction in Acute Pancreatitis (GAMBoost)</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. Model Loading Logic
# -----------------------------------------------------------------------------
@st.cache_resource
def load_gamboost_model(model_path):
  if not os.path.exists(model_path):
    return None
  try:
    model = joblib.load(model_path)
    return model
  except Exception as e:
    st.error(f"Failed to load model: {e}")
    return None


# Sidebar configurations
st.sidebar.header("⚙️ Configuration")
model_file_path = st.sidebar.text_input(
    "Model File Path / Name", value="model_GAMBoost.pkl"
)

model = load_gamboost_model(model_file_path)

if model is None:
  st.sidebar.warning(f"⚠️ Model file '{model_file_path}' not loaded.")
  st.sidebar.info(
      "💡 Please verify `model_GAMBoost.pkl` exists in root repository and"
      " dependencies match."
  )
else:
  st.sidebar.success("✅ GAMBoost Model Successfully Loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Clinical Indicators Info")
st.sidebar.caption(
    "This predictive model incorporates 15 key clinical features selected via"
    " Boruta & RFECV voting algorithms, powered by Generalized Additive"
    " Model Boosting (GAMBoost)."
)

# -----------------------------------------------------------------------------
# 3. Clinical Predictors Input Form
# -----------------------------------------------------------------------------
st.markdown("### 📋 Patient Clinical Parameters Entry")

col1, col2, col3 = st.columns(3)

with col1:
  st.markdown(
      '<div class="sub-header">🩸 Hematology & Inflammation</div>',
      unsafe_allow_html=True,
  )
  hb = st.number_input(
      "Hemoglobin Hb (g/L)", min_value=0.0, max_value=250.0, value=120.0, step=1.0
  )
  wbc = st.number_input(
      "White Blood Cell WBC (×10⁹/L)",
      min_value=0.0,
      max_value=100.0,
      value=9.5,
      step=0.1,
  )
  plt_val = st.number_input(
      "Platelet Count PLT (×10⁹/L)",
      min_value=0.0,
      max_value=1000.0,
      value=200.0,
      step=1.0,
  )
  neutrophil = st.number_input(
      "Neutrophil Percentage (%)",
      min_value=0.0,
      max_value=100.0,
      value=65.0,
      step=0.5,
  )
  lymphocyte = st.number_input(
      "Lymphocyte Count (×10⁹/L)",
      min_value=0.0,
      max_value=20.0,
      value=1.5,
      step=0.1,
  )

with col2:
  st.markdown(
      '<div class="sub-header">🧪 Coagulation, Hepatic & Renal</div>',
      unsafe_allow_html=True,
  )
  inr = st.number_input(
      "International Normalized Ratio (INR)",
      min_value=0.5,
      max_value=10.0,
      value=1.0,
      step=0.01,
  )
  aptt = st.number_input(
      "Activated Partial Thromboplastin Time APTT (s)",
      min_value=10.0,
      max_value=200.0,
      value=32.0,
      step=0.5,
  )
  fibrinogen = st.number_input(
      "Fibrinogen (g/L)",
      min_value=0.0,
      max_value=15.0,
      value=3.0,
      step=0.1,
  )
  tbil = st.number_input(
      "Total Bilirubin TBIL (μmol/L)",
      min_value=0.0,
      max_value=500.0,
      value=15.0,
      step=0.1,
  )
  ast = st.number_input(
      "Aspartate Aminotransferase AST (U/L)",
      min_value=0.0,
      max_value=2000.0,
      value=30.0,
      step=1.0,
  )

with col3:
  st.markdown(
      '<div class="sub-header">🩺 Proteins, Metabolism & Admission</div>',
      unsafe_allow_html=True,
  )
  urea = st.number_input(
      "Urea (mmol/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.1
  )
  cr = st.number_input(
      "Serum Creatinine Cr (μmol/L)",
      min_value=0.0,
      max_value=1500.0,
      value=75.0,
      step=1.0,
  )
  pa = st.number_input(
      "Prealbumin PA (mg/L)",
      min_value=0.0,
      max_value=600.0,
      value=220.0,
      step=1.0,
  )
  onset_to_admit = st.number_input(
      "Onset to Admission Time OnsetToAdmit_H (Hours)",
      min_value=0.0,
      max_value=720.0,
      value=12.0,
      step=1.0,
  )

  patient_source_option = st.radio(
      "Transfer from External Hospital (PatientSource_1)",
      options=[
          "0 - No (Direct Admission / Non-transfer)",
          "1 - Yes (Transferred from External Hospital)",
      ],
      index=0,
      help=(
          "Encoded binary feature indicating whether the patient was"
          " transferred from an outside hospital."
      ),
  )
  patient_source_1 = 1 if "1" in patient_source_option else 0

# -----------------------------------------------------------------------------
# 4. Feature Vector Assembly
# -----------------------------------------------------------------------------
voted_features = [
    "Hb",
    "TBIL",
    "WBC",
    "INR",
    "APTT",
    "Urea",
    "PatientSource_1",
    "Cr",
    "PA",
    "PLT",
    "NeutrophilPercent",
    "LymphocyteCount",
    "OnsetToAdmit_H",
    "Fibrinogen",
    "AST",
]

input_data = pd.DataFrame(
    [[
        hb,
        tbil,
        wbc,
        inr,
        aptt,
        urea,
        patient_source_1,
        cr,
        pa,
        plt_val,
        neutrophil,
        lymphocyte,
        onset_to_admit,
        fibrinogen,
        ast,
    ]],
    columns=voted_features,
)

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. Prediction Execution & Result Rendering
# -----------------------------------------------------------------------------
col_btn, _ = st.columns([1, 4])
with col_btn:
  predict_btn = st.button(
      "🚀 Predict Risk", type="primary", use_container_width=True
  )

if predict_btn:
  if model is None:
    st.error(
        "❌ Unable to load model. Please verify that `model_GAMBoost.pkl` is"
        " present in the repository."
    )
  else:
    try:
      pred_proba = model.predict_proba(input_data)[0][1]

      st.markdown("### 📊 Risk Assessment Results")
      res_col1, res_col2 = st.columns([1, 2])

      # 三分级风险划分
      if pred_proba < 0.15:
        risk_level = "Low Risk"
        delta_color = "normal"
      elif 0.15 <= pred_proba <= 0.45:
        risk_level = "Moderate Risk"
        delta_color = "off"
      else:
        risk_level = "High Risk"
        delta_color = "inverse"

      with res_col1:
        st.metric(
            label="Invasive Ventilation Probability",
            value=f"{pred_proba * 100:.2f}%",
            delta=risk_level,
            delta_color=delta_color,
        )

      with res_col2:
        if pred_proba > 0.45:
          st.markdown(
              f"""
                        <div class="result-box-high">
                            <h3 style="color: #991B1B; margin:0;">🚨 High Risk (> 45%)</h3>
                            <p style="color: #7F1D1D; font-size: 15px; margin-top: 8px;">
                                Estimated probability of requiring <b>invasive mechanical ventilation</b> is <b>{pred_proba*100:.1f}%</b>.<br>
                                Urgent clinical evaluation required. Close monitoring of airway, PaO2/FiO2 ratio, and early ICU involvement is recommended.
                            </p>
                        </div>
                    """,
              unsafe_allow_html=True,
          )
        elif 0.15 <= pred_proba <= 0.45:
          st.markdown(
              f"""
                        <div class="result-box-mid">
                            <h3 style="color: #92400E; margin:0;">⚠️ Moderate Risk (15% – 45%)</h3>
                            <p style="color: #78350F; font-size: 15px; margin-top: 8px;">
                                Estimated probability of requiring <b>invasive mechanical ventilation</b> is <b>{pred_proba*100:.1f}%</b>.<br>
                                Intermediate risk level. Frequent reassessment of respiratory dynamics and oxygenation trends is advised.
                            </p>
                        </div>
                    """,
              unsafe_allow_html=True,
          )
        else:
          st.markdown(
              f"""
                        <div class="result-box-low">
                            <h3 style="color: #166534; margin:0;">✅ Low Risk (< 15%)</h3>
                            <p style="color: #14532D; font-size: 15px; margin-top: 8px;">
                                Estimated probability of requiring <b>invasive mechanical ventilation</b> is <b>{pred_proba*100:.1f}%</b>.<br>
                                Low probability of acute respiratory breakdown. Standard clinical observation recommended.
                            </p>
                        </div>
                    """,
              unsafe_allow_html=True,
          )

      st.markdown("---")

      # -----------------------------------------------------------------------------
      # 6. SHAP Model Interpretability Visualizer
      # -----------------------------------------------------------------------------
      st.markdown("### 🧩 SHAP Patient-Level Explanation")
      st.caption(
          "The plot below illustrates how each clinical predictor contributes"
          " positively or negatively to the predicted risk."
      )

      with st.spinner("Calculating SHAP feature attribution..."):
        try:
          # 针对包含概率预测的模型尝试 Explainer
          explainer = shap.Explainer(model)
          shap_values = explainer(input_data)

          fig, ax = plt.subplots(figsize=(9, 5))
          shap.plots.waterfall(shap_values[0], show=False)
          plt.tight_layout()
          st.pyplot(fig)
          plt.close(fig)
        except Exception as shap_err:
          # 兜底：KernelExplainer
          try:
            explainer = shap.KernelExplainer(model.predict_proba, input_data)
            shap_values = explainer.shap_values(input_data)
            fig, ax = plt.subplots(figsize=(9, 5))
            shap.summary_plot(
                shap_values[1], input_data, plot_type="bar", show=False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
          except Exception as fallback_err:
            st.warning(f"⚠️ Dynamic SHAP rendering error: {shap_err}")

      with st.expander("🔍 View Submitted Feature Array Table"):
        st.dataframe(input_data)

    except Exception as e:
      st.error(f"An error occurred during prediction execution: {e}")
