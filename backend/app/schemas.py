from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict

class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class EthnicityEnum(str, Enum):
    ASIAN = "Asian"
    BLACK = "Black"
    HISPANIC = "Hispanic"
    WHITE = "White"
    OTHER = "Other"

class EducationLevelEnum(str, Enum):
    NO_FORMAL = "No formal"
    HIGHSCHOOL = "Highschool"
    GRADUATE = "Graduate"
    POSTGRADUATE = "Postgraduate"

class IncomeLevelEnum(str, Enum):
    LOW = "Low"
    LOWER_MIDDLE = "Lower-Middle"
    MIDDLE = "Middle"
    UPPER_MIDDLE = "Upper-Middle"
    HIGH = "High"

class EmploymentStatusEnum(str, Enum):
    EMPLOYED = "Employed"
    UNEMPLOYED = "Unemployed"
    RETIRED = "Retired"
    STUDENT = "Student"

class SmokingStatusEnum(str, Enum):
    NEVER = "Never"
    FORMER = "Former"
    CURRENT = "Current"

class PatientInput(BaseModel):
    # Demographics
    age: int = Field(..., ge=1, le=120, description="Patient age in years", example=48)
    gender: GenderEnum = Field(..., description="Gender", example=GenderEnum.FEMALE)
    ethnicity: EthnicityEnum = Field(..., description="Ethnicity group", example=EthnicityEnum.WHITE)
    education_level: EducationLevelEnum = Field(..., description="Highest completed education level", example=EducationLevelEnum.HIGHSCHOOL)
    income_level: IncomeLevelEnum = Field(..., description="Income level bracket", example=IncomeLevelEnum.MIDDLE)
    employment_status: EmploymentStatusEnum = Field(..., description="Current employment status", example=EmploymentStatusEnum.EMPLOYED)

    # Lifestyle & Habits
    smoking_status: SmokingStatusEnum = Field(..., description="Smoking habit status", example=SmokingStatusEnum.FORMER)
    alcohol_consumption_per_week: int = Field(..., ge=0, le=50, description="Alcoholic drinks per week", example=1)
    physical_activity_minutes_per_week: int = Field(..., ge=0, le=1500, description="Minutes of physical activity per week", example=143)
    diet_score: float = Field(..., ge=0.0, le=10.0, description="Diet quality score (0-10)", example=6.7)
    sleep_hours_per_day: float = Field(..., ge=1.0, le=24.0, description="Average sleep hours per day", example=6.5)
    screen_time_hours_per_day: float = Field(..., ge=0.0, le=24.0, description="Daily screen time hours", example=8.7)

    # Medical History (0 = No, 1 = Yes)
    family_history_diabetes: int = Field(..., ge=0, le=1, description="Family history of diabetes (0=No, 1=Yes)", example=0)
    hypertension_history: int = Field(..., ge=0, le=1, description="History of hypertension (0=No, 1=Yes)", example=0)
    cardiovascular_history: int = Field(..., ge=0, le=1, description="History of cardiovascular disease (0=No, 1=Yes)", example=0)

    # Body Measurements & Vitals
    bmi: float = Field(..., ge=10.0, le=80.0, description="Body Mass Index (kg/m²)", example=23.1)
    waist_to_hip_ratio: float = Field(..., ge=0.3, le=2.5, description="Waist-to-hip circumference ratio", example=0.8)
    systolic_bp: int = Field(..., ge=60, le=260, description="Systolic blood pressure (mmHg)", example=129)
    diastolic_bp: int = Field(..., ge=40, le=160, description="Diastolic blood pressure (mmHg)", example=76)
    heart_rate: int = Field(..., ge=40, le=220, description="Resting heart rate (bpm)", example=67)

    # Laboratory Lipid Panel
    cholesterol_total: int = Field(..., ge=50, le=600, description="Total cholesterol (mg/dL)", example=116)
    hdl_cholesterol: int = Field(..., ge=10, le=150, description="HDL cholesterol (mg/dL)", example=55)
    ldl_cholesterol: int = Field(..., ge=20, le=400, description="LDL cholesterol (mg/dL)", example=50)
    triglycerides: int = Field(..., ge=20, le=600, description="Triglycerides (mg/dL)", example=30)

    # Glycemic & Endocrine Indicators
    glucose_fasting: int = Field(..., ge=40, le=400, description="Fasting blood glucose (mg/dL)", example=93)
    glucose_postprandial: int = Field(..., ge=40, le=500, description="Postprandial blood glucose (mg/dL)", example=150)
    insulin_level: float = Field(..., ge=0.0, le=100.0, description="Fasting insulin level (µU/mL)", example=2.0)
    hba1c: float = Field(..., ge=3.0, le=16.0, description="Hemoglobin A1c percentage (%)", example=5.63)

    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "age": 48,
                "gender": "Female",
                "ethnicity": "White",
                "education_level": "Highschool",
                "income_level": "Middle",
                "employment_status": "Employed",
                "smoking_status": "Former",
                "alcohol_consumption_per_week": 1,
                "physical_activity_minutes_per_week": 143,
                "diet_score": 6.7,
                "sleep_hours_per_day": 6.5,
                "screen_time_hours_per_day": 8.7,
                "family_history_diabetes": 0,
                "hypertension_history": 0,
                "cardiovascular_history": 0,
                "bmi": 23.1,
                "waist_to_hip_ratio": 0.8,
                "systolic_bp": 129,
                "diastolic_bp": 76,
                "heart_rate": 67,
                "cholesterol_total": 116,
                "hdl_cholesterol": 55,
                "ldl_cholesterol": 50,
                "triglycerides": 30,
                "glucose_fasting": 93,
                "glucose_postprandial": 150,
                "insulin_level": 2.0,
                "hba1c": 5.63,
            }
        },
    )

class PredictionResponse(BaseModel):
    status: str = Field("success", description="Response status")
    prediction_code: int = Field(..., description="Numeric class prediction (0, 1, 2)")
    prediction_label: str = Field(..., description="Target label: 'No Diabetes', 'Pre-Diabetes', or 'Diabetes'")
    confidence: float = Field(..., description="Confidence probability for predicted class (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Class probability distribution")
    risk_category: str = Field(..., description="Clinical risk category: Low Risk, Moderate Risk, High Risk")
    risk_color: str = Field(..., description="Hex color code associated with risk level")
    key_risk_factors: List[str] = Field(default_factory=list, description="Key clinical and lifestyle risk drivers identified")
    recommendations: List[str] = Field(default_factory=list, description="Tailored lifestyle and medical recommendations")

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    scaler_loaded: bool
    artifacts_directory: str

class MetadataResponse(BaseModel):
    project_name: str
    model_type: str
    best_hyperparameters: Dict
    cv_f1_weighted: float
    test_evaluation: Dict
    target_classes: Dict[str, str]
    mappings: Dict
    feature_importance: List[Dict]
    num_features: int

class PresetsResponse(BaseModel):
    presets: Dict[str, Dict]
