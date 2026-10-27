import json
import logging
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from .config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS_PATH,
    METADATA_PATH,
    PRESETS_PATH,
)
from .schemas import PatientInput, PredictionResponse

logger = logging.getLogger(__name__)


class ModelService:
    _instance: Optional["ModelService"] = None

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns: List[str] = []
        self.metadata: Dict = {}
        self.presets: Dict = {}
        self.target_classes: Dict[int, str] = {
            0: "No Diabetes",
            1: "Pre-Diabetes",
            2: "Diabetes",
        }
        self.load_artifacts()

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_artifacts(self) -> None:
        logger.info("Loading ML artifacts...")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded LightGBM model from {MODEL_PATH}")

        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)
        logger.info(f"Loaded StandardScaler from {SCALER_PATH}")

        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(f"Feature columns file not found at: {FEATURE_COLUMNS_PATH}")
        with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
            self.feature_columns = json.load(f)
        logger.info(f"Loaded {len(self.feature_columns)} feature column definitions")

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                if "target_classes" in self.metadata:
                    self.target_classes = {
                        int(k): v for k, v in self.metadata["target_classes"].items()
                    }
        else:
            logger.warning(f"Metadata file not found at {METADATA_PATH}, using defaults.")

        if PRESETS_PATH.exists():
            with open(PRESETS_PATH, "r", encoding="utf-8") as f:
                self.presets = json.load(f)
        else:
            logger.warning(f"Presets file not found at {PRESETS_PATH}")

    def transform_input(self, patient_data: PatientInput) -> np.ndarray:
        data_dict = patient_data.model_dump()
        df = pd.DataFrame([data_dict])

        # Map Ordinal features
        education_map = self.metadata.get("mappings", {}).get(
            "education_level",
            {"No formal": 0, "Highschool": 1, "Postgraduate": 2, "Graduate": 3},
        )
        income_map = self.metadata.get("mappings", {}).get(
            "income_level",
            {"Low": 0, "Lower-Middle": 1, "Middle": 2, "Upper-Middle": 3, "High": 4},
        )

        df["education_level"] = df["education_level"].map(education_map)
        df["income_level"] = df["income_level"].map(income_map)

        # One-Hot Encoding on nominal columns
        cols_onehot = ["gender", "ethnicity", "employment_status", "smoking_status"]
        df_encoded = pd.get_dummies(df, columns=cols_onehot, drop_first=True)

        # Align with the 35 feature columns expected by the model
        df_aligned = df_encoded.reindex(columns=self.feature_columns, fill_value=0)

        # Apply Standard scaling
        scaled_features = self.scaler.transform(df_aligned)
        return scaled_features

    def _analyze_clinical_insights(
        self, patient: PatientInput, pred_code: int
    ) -> Tuple[List[str], List[str]]:
        risk_factors: List[str] = []
        recommendations: List[str] = []

        # Glycemic indicators
        if patient.glucose_fasting >= 126:
            risk_factors.append(f"Elevated fasting glucose ({patient.glucose_fasting} mg/dL, threshold ≥ 126 mg/dL)")
        elif patient.glucose_fasting >= 100:
            risk_factors.append(f"Impaired fasting glucose ({patient.glucose_fasting} mg/dL, pre-diabetic range 100-125 mg/dL)")

        if patient.hba1c >= 6.5:
            risk_factors.append(f"Elevated HbA1c ({patient.hba1c}%, diagnostic diabetic threshold ≥ 6.5%)")
        elif patient.hba1c >= 5.7:
            risk_factors.append(f"Pre-diabetic HbA1c range ({patient.hba1c}%, range 5.7% - 6.4%)")

        if patient.glucose_postprandial >= 200:
            risk_factors.append(f"Elevated 2-hour postprandial glucose ({patient.glucose_postprandial} mg/dL ≥ 200 mg/dL)")

        # Physical & Metabolic factors
        if patient.bmi >= 30.0:
            risk_factors.append(f"Classified as Obese by BMI ({patient.bmi:.1f} kg/m² ≥ 30.0)")
        elif patient.bmi >= 25.0:
            risk_factors.append(f"Overweight BMI range ({patient.bmi:.1f} kg/m²)")

        if patient.physical_activity_minutes_per_week < 150:
            risk_factors.append(
                f"Sedentary/Low physical activity ({patient.physical_activity_minutes_per_week} mins/week vs WHO recommendation of ≥ 150 mins)"
            )

        if patient.family_history_diabetes == 1:
            risk_factors.append("Positive genetic predisposition (family history of diabetes)")

        if patient.systolic_bp >= 140 or patient.diastolic_bp >= 90:
            risk_factors.append(f"Elevated blood pressure ({patient.systolic_bp}/{patient.diastolic_bp} mmHg)")

        if patient.smoking_status in ["Current", "Former"]:
            risk_factors.append(f"Smoking history ({patient.smoking_status}) increases vascular & metabolic risk")

        # Recommendations based on predicted stage
        if pred_code == 2:  # Diabetes
            recommendations.append("Immediate comprehensive clinical consultation with an endocrinologist or primary care physician.")
            recommendations.append("Undergo confirmatory laboratory tests (repeat fasting plasma glucose, HbA1c, and oral glucose tolerance test).")
            recommendations.append("Adopt a medically supervised personalized nutritional plan with carbohydrate monitoring.")
            recommendations.append("Target a minimum of 150 minutes of moderate-intensity aerobic exercise per week as tolerated.")
            recommendations.append("Establish routine self-monitoring of blood glucose levels.")
        elif pred_code == 1:  # Pre-Diabetes
            recommendations.append("Proactive lifestyle intervention can prevent or delay progression to Type 2 diabetes by up to 58%.")
            recommendations.append("Aim for a realistic 5% to 7% body weight reduction if currently overweight or obese.")
            recommendations.append("Engage in brisk walking or equivalent aerobic exercise for at least 30 minutes, 5 days per week.")
            recommendations.append("Shift toward a Mediterranean or dietary fiber-rich eating pattern with low glycemic index foods.")
            recommendations.append("Schedule follow-up glycemic screening (HbA1c / fasting glucose) every 6 to 12 months.")
        else:  # No Diabetes
            recommendations.append("Maintain your current healthy lifestyle practices and well-balanced dietary habits.")
            recommendations.append("Continue regular physical activity of at least 150 minutes per week.")
            recommendations.append("Maintain optimal body weight and cardiovascular health.")
            recommendations.append("Conduct routine wellness checkups and glycemic screenings annually.")

        return risk_factors, recommendations

    def predict(self, patient_data: PatientInput) -> PredictionResponse:
        scaled_features = self.transform_input(patient_data)

        # Model inferences
        pred_code = int(self.model.predict(scaled_features)[0])
        prob_array = self.model.predict_proba(scaled_features)[0]

        pred_label = self.target_classes.get(pred_code, "Unknown")
        confidence = round(float(prob_array[pred_code]), 4)

        probabilities = {
            self.target_classes.get(i, f"Class {i}"): round(float(prob_array[i]), 4)
            for i in range(len(self.target_classes))
        }

        # Risk categorization & visual cues
        if pred_code == 2:
            risk_category = "High Risk"
            risk_color = "#E74C3C"  # Red
        elif pred_code == 1:
            risk_category = "Moderate Risk"
            risk_color = "#F39C12"  # Amber / Orange
        else:
            risk_category = "Low Risk"
            risk_color = "#2ECC71"  # Emerald Green

        key_risk_factors, recommendations = self._analyze_clinical_insights(
            patient_data, pred_code
        )

        return PredictionResponse(
            status="success",
            prediction_code=pred_code,
            prediction_label=pred_label,
            confidence=confidence,
            probabilities=probabilities,
            risk_category=risk_category,
            risk_color=risk_color,
            key_risk_factors=key_risk_factors,
            recommendations=recommendations,
        )

# Global singleton helper
def get_model_service() -> ModelService:
    return ModelService.get_instance()
