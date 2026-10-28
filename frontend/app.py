import os
import json
from pathlib import Path
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PRESETS_FILE = PROJECT_ROOT / "notebook" / "artifacts" / "presets.json"

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Diabetes Prediction Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================
# CUSTOM CSS INJECTION
# ==========================================
st.markdown(
    """
    <style>
        .hero-title {
            text-align: center;
            font-size: 3.1rem;
            font-weight: 800;
            margin-bottom: 0px;
            padding-top: 1.5rem;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            text-align: center;
            font-size: 1.15rem;
            color: #555;
            margin-top: 15px;
            margin-bottom: 35px;
            padding: 0 12%;
            line-height: 1.6;
        }

        .stButton>button[kind="primary"] {
            background-color: #0072ce;
            border-color: #0072ce;
            color: #fff;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.55rem 1.2rem;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #005fa3;
            border-color: #005fa3;
            box-shadow: 0 4px 12px rgba(0, 114, 206, 0.28);
        }
        
        .challenge-card {
            background-color: #f8f9fa;
            padding: 2.2rem;
            border-radius: 12px;
            border-left: 6px solid #0072ce;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .challenge-card h4 {
            margin-top: 1.2rem;
            margin-bottom: 0.4rem;
            color: #0072ce;
            font-weight: 700;
        }
        
        .tech-card {
            background-color: #f1f3f6;
            padding: 1.5rem;
            border-radius: 10px;
            height: 100%;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .tech-card h4 {
            color: #0072ce;
            margin-top: 0;
            margin-bottom: 0.8rem;
        }
        .tech-card ul {
            padding-left: 1.2rem;
            margin-bottom: 0;
        }
        
        .result-box-safe {
            padding: 22px;
            border-radius: 12px;
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            margin-bottom: 15px;
        }
        .result-box-warning {
            padding: 22px;
            border-radius: 12px;
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
            margin-bottom: 15px;
        }
        .result-box-danger {
            padding: 22px;
            border-radius: 12px;
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            margin-bottom: 15px;
        }
        .result-box-neutral {
            padding: 35px 20px;
            border-radius: 12px;
            background-color: #f8f9fa;
            color: #6c757d;
            border: 1px dashed #ced4da;
            text-align: center;
            font-size: 1rem;
        }
        
        .metric-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 6px;
        }
            
        .footer {
            text-align: center;
            padding: 25px 20px;
            color: #666;
            border-top: 1px solid #eaeaea;
            margin-top: 50px;
        }
        .footer a {
            color: #0072ce;
            text-decoration: none;
            margin: 0 10px;
            font-weight: 500;
        }
        .footer a:hover {
            text-decoration: underline;
        }

        @media (prefers-color-scheme: dark) {
            .hero-subtitle {
                color: #aaaaaa;
            }
            .challenge-card {
                background-color: #1e1e1e;
                border-left: 6px solid #38bdf8;
            }
            .challenge-card h4 {
                color: #38bdf8;
            }
            .tech-card {
                background-color: #262730;
                border-color: rgba(255,255,255,0.08);
            }
            .tech-card h4 {
                color: #38bdf8;
            }
            .result-box-neutral {
                background-color: #1e1e1e;
                color: #999;
                border-color: #444;
            }
            .footer {
                border-top-color: #333;
                color: #888;
            }
            .footer a {
                color: #38bdf8;
            }
        }
    </style>
""", unsafe_allow_html=True)

def load_presets_data():
    try:
        resp = requests.get(f"{API_URL}/api/presets", timeout=2)
        if resp.status_code == 200:
            return resp.json().get("presets", {})
    except Exception:
        pass

    if PRESETS_FILE.exists():
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


presets_dict = load_presets_data()

# ==========================================
# SECTION 1: HERO
# ==========================================
st.markdown(
    '<h1 class="hero-title">Diabetes Prediction Analytics</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-subtitle">An End-to-End Machine Learning System powered by an optimized LightGBM Classifier '
    "to proactively screen and detect diabetes stages (No Diabetes, Pre-Diabetes, and Diabetes) "
    "based on 28 comprehensive clinical, laboratory, and lifestyle indicators.</p>",
    unsafe_allow_html=True,
)

col_hero1, col_hero2, col_hero3, col_hero4 = st.columns([2.2, 1.5, 1.7, 2.2])

with col_hero2:
    st.markdown(
        """
        <a href="#interactive-screening" target="_self" style="text-decoration: none;">
            <div style="
                background-color: #0072ce; 
                color: white; 
                padding: 0.5rem 1.2rem; 
                border-radius: 0.5rem; 
                text-align: center; 
                font-weight: 600;
                border: 1px solid #0072ce;
                box-shadow: 0 2px 6px rgba(0, 114, 206, 0.25);
            ">
                Start Simulation
            </div>
        </a>
        """, unsafe_allow_html=True,
    )

with col_hero3:
    st.link_button(
        "View Research Notebook",
        url="https://colab.research.google.com/drive/1XQJaMvqcu8-feQhKdKKJ9hofLjuDr286?usp=sharing",
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ==========================================
# SECTION 2: PROJECT BACKGROUND
# ==========================================
st.header("Project Background")

st.markdown("""
<div class="challenge-card">
    <p>Diabetes Mellitus represents one of the fastest growing global chronic health crises. Conventional medical protocols 
    frequently rely on reactive management diagnosing conditions only after metabolic damage has manifested. 
    This project introduces a proactive, data-driven machine learning solution for preemptive risk stratification. 
    The primary research and clinical objectives are:</p>
    <h4>1. Early Interception of Pre-Diabetes</h4>
    <p>Pre-diabetes is a reversible window of opportunity. Clinical evidence from the American Diabetes Association (ADA) 
    confirms that proactive lifestyle modifications can reduce progression to Type 2 diabetes by up to 58%. 
    Accurately isolating pre-diabetic patients prevents costly long-term vascular complications.</p>
    <h4>2. Holistic Clinical & Behavioral Synthesis</h4>
    <p>Integrating biological markers (HbA1c, fasting and postprandial glucose, insulin) with patient habits 
    (physical activity, sleep hygiene, screen time, diet quality) to provide a complete metabolic risk profile.</p>
    <h4>3. Zero Tolerance for False Negatives</h4>
    <p>Trained and validated on 100,000 diverse patient records, our tuned LightGBM model prioritizes clinical Recall 
    (achieving 1.00 recall on Pre-Diabetes and No Diabetes), ensuring potential cases are never overlooked.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SECTION 3: THE ENGINE (TECH STACK)
# ==========================================
st.header("The Engine (Tech Stack)")
st.markdown(
    "A full-stack, decoupled architecture designed for rapid inference, reliability, and clinical interpretability."
)

col_t1, col_t2, col_t3, col_t4 = st.columns(4)

with col_t1:
    st.markdown(
        """
    <div class="tech-card">
        <h4>Backend</h4>
        <ul>
            <li><b>FastAPI</b> & <b>Uvicorn</b></li>
            <li>Async REST API Service</li>
            <li>Pydantic v2 Boundary Validation</li>
            <li>Singleton Model Loading</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_t2:
    st.markdown(
        """
    <div class="tech-card">
        <h4>Frontend</h4>
        <ul>
            <li><b>Streamlit</b></li>
            <li>Interactive Clinical UI</li>
            <li>1-Click Patient Presets</li>
            <li>Dynamic Risk Categorization</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_t3:
    st.markdown(
        """
    <div class="tech-card">
        <h4>ML Framework</h4>
        <ul>
            <li><b>LightGBM Classifier</b></li>
            <li>Scikit-Learn Preprocessing</li>
            <li>GridSearchCV 5-Fold Tuning</li>
            <li>Joblib Binary Serialization</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col_t4:
    st.markdown(
        """
    <div class="tech-card">
        <h4>Data & Tools</h4>
        <ul>
            <li>100,000 Patient Records</li>
            <li>Pandas, NumPy, Pillow</li>
            <li>StandardScaler Scaling</li>
            <li>Matplotlib & Seaborn EDA</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ==========================================
# SECTION 4: LIVE DEMO (Interactive Simulator)
# ==========================================
st.markdown('<div id="interactive-screening"></div>', unsafe_allow_html=True)
st.header("Interactive Patient Screening")
st.markdown(
    "Simulate patient risk evaluation in real-time. Select a pre-configured patient profile "
    "or input custom clinical measurements below."
)

# Preset Selector
preset_names = ["-- Select Preset Profile --"] + list(
    presets_dict.keys()
)
selected_preset = st.selectbox(
    "Load Preset Patient Profile:", preset_names, index=0
)

# Active profile data
active_data = {}
if selected_preset != preset_names[0] and selected_preset in presets_dict:
    active_data = presets_dict[selected_preset]
    st.success(
        f"Loaded **{selected_preset}** data. You can inspect or tweak any parameters below before running analysis."
    )

st.markdown("#### Patient Parameters")

col_left, col_right = st.columns(2)

with col_left:
    with st.expander("Demographic Indicators", expanded=True):
        cd1, cd2 = st.columns(2)
        with cd1:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=int(active_data.get("age", 48)),
            )
            gender_options = ["Female", "Male", "Other"]
            gender_default = active_data.get("gender", "Female")
            gender = st.selectbox(
                "Gender",
                gender_options,
                index=(
                    gender_options.index(gender_default)
                    if gender_default in gender_options
                    else 0
                ),
            )
            ethnicity_options = ["White", "Asian", "Black", "Hispanic", "Other"]
            ethnicity_default = active_data.get("ethnicity", "White")
            ethnicity = st.selectbox(
                "Ethnicity",
                ethnicity_options,
                index=(
                    ethnicity_options.index(ethnicity_default)
                    if ethnicity_default in ethnicity_options
                    else 0
                ),
            )
        with cd2:
            edu_options = [
                "Highschool",
                "Graduate",
                "Postgraduate",
                "No formal",
            ]
            edu_default = active_data.get("education_level", "Highschool")
            education_level = st.selectbox(
                "Education Level",
                edu_options,
                index=(
                    edu_options.index(edu_default)
                    if edu_default in edu_options
                    else 0
                ),
            )
            inc_options = [
                "Middle",
                "Upper-Middle",
                "Lower-Middle",
                "High",
                "Low",
            ]
            inc_default = active_data.get("income_level", "Middle")
            income_level = st.selectbox(
                "Income Level",
                inc_options,
                index=(
                    inc_options.index(inc_default)
                    if inc_default in inc_options
                    else 0
                ),
            )
            emp_options = ["Employed", "Unemployed", "Retired", "Student"]
            emp_default = active_data.get("employment_status", "Employed")
            employment_status = st.selectbox(
                "Employment Status",
                emp_options,
                index=(
                    emp_options.index(emp_default)
                    if emp_default in emp_options
                    else 0
                ),
            )

    with st.expander("Lifestyle Habits", expanded=True):
        cl1, cl2 = st.columns(2)
        with cl1:
            smoke_options = ["Never", "Former", "Current"]
            smoke_default = active_data.get("smoking_status", "Former")
            smoking_status = st.selectbox(
                "Smoking Status",
                smoke_options,
                index=(
                    smoke_options.index(smoke_default)
                    if smoke_default in smoke_options
                    else 0
                ),
            )
            alcohol = st.number_input(
                "Alcohol Drinks / Week",
                min_value=0,
                max_value=50,
                value=int(active_data.get("alcohol_consumption_per_week", 1)),
            )
            activity = st.number_input(
                "Physical Activity (minutes/week)",
                min_value=0,
                max_value=1500,
                value=int(
                    active_data.get("physical_activity_minutes_per_week", 143)
                ),
            )
        with cl2:
            diet = st.slider(
                "Diet Quality Score (0-10)",
                min_value=0.0,
                max_value=10.0,
                value=float(active_data.get("diet_score", 6.7)),
                step=0.1,
            )
            sleep = st.slider(
                "Sleep (hours/day)",
                min_value=1.0,
                max_value=24.0,
                value=float(active_data.get("sleep_hours_per_day", 6.5)),
                step=0.1,
            )
            screen = st.slider(
                "Screen Time (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=float(active_data.get("screen_time_hours_per_day", 8.7)),
                step=0.1,
            )

    with st.expander("Medical History", expanded=True):
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            family_hist = st.radio(
                "Family History of Diabetes",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=int(active_data.get("family_history_diabetes", 0)),
            )
        with cm2:
            hypertension = st.radio(
                "Hypertension History",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=int(active_data.get("hypertension_history", 0)),
            )
        with cm3:
            cardiovascular = st.radio(
                "Cardiovascular History",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=int(active_data.get("cardiovascular_history", 0)),
            )

with col_right:
    with st.expander("Clinical & Laboratory Measurements", expanded=True):
        cc1, cc2 = st.columns(2)
        with cc1:
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0,
                max_value=80.0,
                value=float(active_data.get("bmi", 23.1)),
                step=0.1,
            )
            waist_hip = st.number_input(
                "Waist-to-Hip Ratio",
                min_value=0.3,
                max_value=2.5,
                value=float(active_data.get("waist_to_hip_ratio", 0.8)),
                step=0.01,
            )
            systolic = st.number_input(
                "Systolic BP (mmHg)",
                min_value=60,
                max_value=260,
                value=int(active_data.get("systolic_bp", 129)),
            )
            diastolic = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=40,
                max_value=160,
                value=int(active_data.get("diastolic_bp", 76)),
            )
            heart_rate = st.number_input(
                "Resting Heart Rate (bpm)",
                min_value=40,
                max_value=220,
                value=int(active_data.get("heart_rate", 67)),
            )
        with cc2:
            cholesterol = st.number_input(
                "Total Cholesterol (mg/dL)",
                min_value=50,
                max_value=600,
                value=int(active_data.get("cholesterol_total", 116)),
            )
            hdl = st.number_input(
                "HDL Cholesterol (mg/dL)",
                min_value=10,
                max_value=150,
                value=int(active_data.get("hdl_cholesterol", 55)),
            )
            ldl = st.number_input(
                "LDL Cholesterol (mg/dL)",
                min_value=20,
                max_value=400,
                value=int(active_data.get("ldl_cholesterol", 50)),
            )
            triglycerides = st.number_input(
                "Triglycerides (mg/dL)",
                min_value=20,
                max_value=600,
                value=int(active_data.get("triglycerides", 30)),
            )

        st.markdown(
            "**Primary Glycemic Biomarkers (Highest Feature Importance)**"
        )
        cg1, cg2, cg3 = st.columns(3)
        with cg1:
            glucose_fasting = st.number_input(
                "Fasting Glucose (mg/dL)",
                min_value=40,
                max_value=400,
                value=int(active_data.get("glucose_fasting", 93)),
                help="Fasting blood sugar level. Normal: <100, Pre-diabetes: 100-125, Diabetes: ≥126 mg/dL",
            )
        with cg2:
            glucose_post = st.number_input(
                "Postprandial Glucose (mg/dL)",
                min_value=40,
                max_value=500,
                value=int(active_data.get("glucose_postprandial", 150)),
                help="2-hour postprandial glucose. Normal: <140, Pre-diabetes: 140-199, Diabetes: ≥200 mg/dL",
            )
        with cg3:
            hba1c = st.number_input(
                "HbA1c (%)",
                min_value=3.0,
                max_value=16.0,
                value=float(active_data.get("hba1c", 5.63)),
                step=0.01,
                help="Long-term blood sugar index. Normal: <5.7%, Pre-diabetes: 5.7%-6.4%, Diabetes: ≥6.5%",
            )
        insulin = st.number_input(
            "Fasting Insulin (µU/mL)",
            min_value=0.0,
            max_value=100.0,
            value=float(active_data.get("insulin_level", 2.0)),
            step=0.1,
        )

st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1.2, 1])
with col_btn2:
    analyze_button = st.button(
        "Analyze Patient Risk", type="primary", use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Live Analysis Result")

if analyze_button:
    patient_payload = {
        "age": int(age),
        "gender": str(gender),
        "ethnicity": str(ethnicity),
        "education_level": str(education_level),
        "income_level": str(income_level),
        "employment_status": str(employment_status),
        "smoking_status": str(smoking_status),
        "alcohol_consumption_per_week": int(alcohol),
        "physical_activity_minutes_per_week": int(activity),
        "diet_score": float(diet),
        "sleep_hours_per_day": float(sleep),
        "screen_time_hours_per_day": float(screen),
        "family_history_diabetes": int(family_hist),
        "hypertension_history": int(hypertension),
        "cardiovascular_history": int(cardiovascular),
        "bmi": float(bmi),
        "waist_to_hip_ratio": float(waist_hip),
        "systolic_bp": int(systolic),
        "diastolic_bp": int(diastolic),
        "heart_rate": int(heart_rate),
        "cholesterol_total": int(cholesterol),
        "hdl_cholesterol": int(hdl),
        "ldl_cholesterol": int(ldl),
        "triglycerides": int(triglycerides),
        "glucose_fasting": int(glucose_fasting),
        "glucose_postprandial": int(glucose_post),
        "insulin_level": float(insulin),
        "hba1c": float(hba1c),
    }

    with st.spinner("Processing metabolic vectors and querying model..."):
        try:
            response = requests.post(
                f"{API_URL}/api/predict", json=patient_payload, timeout=5
            )
            if response.status_code == 200:
                res = response.json()
                label = res.get("prediction_label", "Unknown")
                confidence = res.get("confidence", 0.0)
                risk_cat = res.get("risk_category", "Unknown")
                probs = res.get("probabilities", {})
                risk_factors = res.get("key_risk_factors", [])
                recs = res.get("recommendations", [])

                col_res1, col_res2 = st.columns([1, 1])

                with col_res1:
                    if label == "No Diabetes":
                        st.markdown(
                            f"""
                        <div class="result-box-safe">
                            <h3 style="margin-top:0; color:#155724; font-weight:800;">🟢 {label.upper()}</h3>
                            <p style="margin:4px 0;"><b>Confidence Score:</b> {confidence*100:.1f}%</p>
                            <p style="margin:4px 0;"><b>Risk Status:</b> {risk_cat} (Low Glycemic Risk)</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    elif label == "Pre-Diabetes":
                        st.markdown(
                            f"""
                        <div class="result-box-warning">
                            <h3 style="margin-top:0; color:#856404; font-weight:800;">🟡 {label.upper()}</h3>
                            <p style="margin:4px 0;"><b>Confidence Score:</b> {confidence*100:.1f}%</p>
                            <p style="margin:4px 0;"><b>Risk Status:</b> {risk_cat} (Actionable Window)</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                        <div class="result-box-danger">
                            <h3 style="margin-top:0; color:#721c24; font-weight:800;">🔴 {label.upper()}</h3>
                            <p style="margin:4px 0;"><b>Confidence Score:</b> {confidence*100:.1f}%</p>
                            <p style="margin:4px 0;"><b>Risk Status:</b> {risk_cat} (Clinical Attention Recommended)</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("##### Stage Probability Distribution")
                    for cls_name, prob_val in probs.items():
                        prob_pct = prob_val * 100
                        st.write(f"**{cls_name}**: `{prob_pct:.1f}%`")
                        st.progress(float(prob_val))

                with col_res2:
                    if risk_factors:
                        st.markdown("##### ⚠️ Identified Risk Drivers")
                        for rf in risk_factors:
                            st.markdown(f"- {rf}")

                    if recs:
                        st.markdown("##### 💡 Clinical & Lifestyle Guidance")
                        for r in recs:
                            st.markdown(f"- {r}")

            elif response.status_code == 422:
                st.error("Validation Error: Please verify all input ranges.")
                st.json(response.json())
            else:
                st.error(
                    f"API Error: HTTP {response.status_code} - {response.text}"
                )
        except requests.exceptions.ConnectionError:
            st.error("🚨 Failed to connect to FastAPI Backend!")
            st.info(
                f"Please make sure the FastAPI server is running on `{API_URL}`."
            )
else:
    st.markdown(
        """
    <div class="result-box-neutral">
        <b>Waiting for patient data...</b><br>
        Select a preset profile from the dropdown above or customize parameters, 
        then click <b>"Analyze Patient Risk"</b> to generate real-time screening results.
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# ==========================================
# SECTION 5: DATA EXTRACTION & EDA
# ==========================================
st.header("Data Extraction & Visualizations (EDA)")
st.markdown(
    "Key exploratory findings from analyzing 100,000 multi-ethnic patient health records."
)

col_eda_text, col_eda_img = st.columns([1, 1.2])

with col_eda_text:
    st.subheader("Feature Correlations & Outliers")
    st.markdown("""
    - **Clean Baseline**: Zero missing values across all 100,000 rows and 31 initial columns, confirming exceptional data hygiene.
    - **High Glycemic Correlation**: `hba1c` demonstrates a high linear correlation with `glucose_postprandial` (0.93) and `glucose_fasting` (0.70), confirming biological cohesion.
    - **Physical Anthropometry**: Strong correlation between `waist_to_hip_ratio` and `bmi` (0.77).
    - **IQR Outlier Quantification**: `physical_activity_minutes_per_week` exhibits the highest outlier proportion (3.20%), representing individuals with intense physical training regimens. These points were preserved as valid clinical signals.
    """)

with col_eda_img:
    corr_img_path = os.path.join(BASE_DIR, "assets", "correlation.png")
    if os.path.exists(corr_img_path):
        st.image(
            corr_img_path,
            caption="Correlation Heatmap across Numerical Health Markers",
            use_container_width=True,
        )
    else:
        st.info("Correlation graph will be displayed here.")

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ==========================================
# SECTION 6: MODEL BENCHMARKING & IMPORTANCE
# ==========================================
st.header("Model Benchmarking & Feature Importance")
st.markdown(
    "Comparative evaluation demonstrating why LightGBM delivers optimal clinical screening reliability."
)

col_bench_table, col_bench_img = st.columns([1, 1.2])

with col_bench_table:
    st.subheader("Algorithm Performance Comparison")
    st.markdown(
        "Four algorithms were rigorously benchmarked on the 20,000-sample unseen test set:"
    )

    st.markdown("""
    | Model Architecture | Accuracy | Weighted F1 | Critical Recall |
    | :--- | :---: | :---: | :---: |
    | **LightGBM Classifier (Tuned)** | **91.84%** | **0.92** | **1.00** |
    | **XGBoost Classifier** | 91.62% | 0.92 | 0.99 |
    | **Logistic Regression** | 82.17% | 0.82 | 0.73 |
    | **K-Nearest Neighbors** | 72.76% | 0.73 | 0.64 |
    """)

    st.markdown("""
    > **Why LightGBM was selected as the final production model:**  
    > In medical risk screening, a false negative (failing to identify an individual with pre-diabetes or diabetes) 
    > carries far higher clinical risks than a false positive. LightGBM achieved **1.00 Recall** on both 
    > Pre-Diabetes and No Diabetes classes while maintaining 99.9% precision on confirmed Diabetes cases.
    """)

with col_bench_img:
    feat_img_path = os.path.join(BASE_DIR, "assets", "feature_importance.png")
    if os.path.exists(feat_img_path):
        st.image(
            feat_img_path,
            caption="Top Feature Importances from Optimized LightGBM Classifier",
            use_container_width=True,
        )
    else:
        st.info("Feature Importance chart will be displayed here.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.divider()

# ==========================================
# SECTION 7: PROJECT ORIGIN & CERTIFICATION
# ==========================================
st.header("Project Origin & Certification")
st.markdown(
    "Demonstrates foundational machine learning competencies, rigorous evaluation methodology, and end-to-end deployment."
)

col_cert_img, col_cert_text = st.columns([1, 1.5])

with col_cert_img:
    cert_path = os.path.join(BASE_DIR, "assets", "certificate.png")
    if os.path.exists(cert_path):
        st.image(cert_path, use_container_width=True)
    else:
        st.info(
            "🎓 Certificate image will be displayed here. Please place `certificate.png` in `frontend/assets/`."
        )

with col_cert_text:
    st.markdown("""
    This project was built as an end-to-end Machine Learning Capstone for predictive analytics and clinical risk classification. 
    It incorporates rigorous data exploration, outlier quantification, data leakage prevention, stratified cross-validation, 
    hyperparameter tuning, and containerized REST API deployment.
    """)
    st.markdown("<br>", unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        st.link_button(
            "View Course Curriculum",
            "https://www.dicoding.com/academies/319",
            use_container_width=True,
        )
    with btn_col2:
        st.link_button(
            "Verify Developer Credential",
            "https://www.dicoding.com/certificates/L4PQ25D0OZO1",
            use_container_width=True,
        )
    with btn_col3:
        st.link_button(
            "Read Submission Report",
            "https://github.com/gentahape/prediksi-diabetes-berdasarkan-indikator-kesehatan-dan-gaya-hidup",
            use_container_width=True,
        )

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# SECTION 8: FOOTER & PROJECT META
# ==========================================
st.markdown(
    """
    <div class="footer">
        <p>Built by <b>Genta Haetami Putra</b> — Fullstack Web Developer & ML Enthusiast</p>
        <p>
            <a href="https://gentahp.my.id" target="_blank">Personal Website</a> | 
            <a href="https://linkedin.com/in/gentahp" target="_blank">LinkedIn</a> | 
            <a href="https://github.com/gentahape/diabetes-predictive-analytics" target="_blank">GitHub Repository</a> | 
            <a href="https://colab.research.google.com/drive/1XQJaMvqcu8-feQhKdKKJ9hofLjuDr286?usp=sharing" target="_blank">Google Colab Research</a>
        </p>
    </div>
""",
    unsafe_allow_html=True,
)
