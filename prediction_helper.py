import os
import joblib
import pandas as pd

def load_model(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    return joblib.load(file_path)

# Define paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Ensure the artifacts directory exists
if not os.path.exists(ARTIFACTS_DIR):
    raise FileNotFoundError(f"Artifacts directory not found: {ARTIFACTS_DIR}")

model_young_path = os.path.join(ARTIFACTS_DIR, "model_young.joblib")
model_rest_path = os.path.join(ARTIFACTS_DIR, "model_rest.joblib")
scaler_young_path = os.path.join(ARTIFACTS_DIR, "scaler_young.joblib")
scaler_rest_path = os.path.join(ARTIFACTS_DIR, "scaler_rest.joblib")

# Load models and scalers
model_young = load_model(model_young_path)
model_rest = load_model(model_rest_path)
scaler_young = load_model(scaler_young_path)
scaler_rest = load_model(scaler_rest_path)

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    return total_risk_score / 14  # Normalize using max score 14

def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Assign categorical values
    if input_dict.get('Gender') == 'Male':
        df['gender_Male'] = 1
    if input_dict.get('Region') in ['Northwest', 'Southeast', 'Southwest']:
        df[f"region_{input_dict['Region']}"] = 1
    if input_dict.get('Marital Status') == 'Unmarried':
        df['marital_status_Unmarried'] = 1
    if input_dict.get('BMI Category') in ['Obesity', 'Overweight', 'Underweight']:
        df[f"bmi_category_{input_dict['BMI Category']}"] = 1
    if input_dict.get('Smoking Status') in ['Occasional', 'Regular']:
        df[f"smoking_status_{input_dict['Smoking Status']}"] = 1
    if input_dict.get('Employment Status') in ['Salaried', 'Self-Employed']:
        df[f"employment_status_{input_dict['Employment Status']}"] = 1

    # Assign numerical values
    df['insurance_plan'] = insurance_plan_encoding.get(input_dict.get('Insurance Plan'), 1)
    df['age'] = input_dict.get('Age', 0)
    df['number_of_dependants'] = input_dict.get('Number of Dependants', 0)
    df['income_lakhs'] = input_dict.get('Income in Lakhs', 0)
    df['genetical_risk'] = input_dict.get("Genetical Risk", 0)
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'none'))

    # Apply scaling
    df = handle_scaling(input_dict.get('Age', 0), df)
    return df

def handle_scaling(age, df):
    scaler_object = scaler_young if age <= 25 else scaler_rest
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    # Apply scaling
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

def predict(input_dict):
    input_df = preprocess_input(input_dict)
    model = model_young if input_dict.get('Age', 0) <= 25 else model_rest
    prediction = model.predict(input_df)
    return int(prediction[0])

