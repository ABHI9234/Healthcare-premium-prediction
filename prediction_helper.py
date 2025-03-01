# # codebasics ML course: codebasics.io, all rights reserverd
#
# import pandas as pd
# import joblib
#
# model_young = joblib.load("artifacts\model_young.joblib")
# model_rest = joblib.load("artifacts\model_rest.joblib")
# scaler_young = joblib.load("artifacts\scaler_young.joblib")
# scaler_rest = joblib.load("artifacts\scaler_rest.joblib")
#
# def calculate_normalized_risk(medical_history):
#     risk_scores = {
#         "diabetes": 6,
#         "heart disease": 8,
#         "high blood pressure": 6,
#         "thyroid": 5,
#         "no disease": 0,
#         "none": 0
#     }
#     # Split the medical history into potential two parts and convert to lowercase
#     diseases = medical_history.lower().split(" & ")
#
#     # Calculate the total risk score by summing the risk scores for each part
#     total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found
#
#     max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
#     min_score = 0  # Since the minimum score is always 0
#
#     # Normalize the total risk score
#     normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
#
#     return normalized_risk_score
#
# def preprocess_input(input_dict):
#     # Define the expected columns and initialize the DataFrame with zeros
#     expected_columns = [
#         'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
#         'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
#         'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
#         'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
#     ]
#
#     insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
#
#     df = pd.DataFrame(0, columns=expected_columns, index=[0])
#     # df.fillna(0, inplace=True)
#
#     # Manually assign values for each categorical input based on input_dict
#     for key, value in input_dict.items():
#         if key == 'Gender' and value == 'Male':
#             df['gender_Male'] = 1
#         elif key == 'Region':
#             if value == 'Northwest':
#                 df['region_Northwest'] = 1
#             elif value == 'Southeast':
#                 df['region_Southeast'] = 1
#             elif value == 'Southwest':
#                 df['region_Southwest'] = 1
#         elif key == 'Marital Status' and value == 'Unmarried':
#             df['marital_status_Unmarried'] = 1
#         elif key == 'BMI Category':
#             if value == 'Obesity':
#                 df['bmi_category_Obesity'] = 1
#             elif value == 'Overweight':
#                 df['bmi_category_Overweight'] = 1
#             elif value == 'Underweight':
#                 df['bmi_category_Underweight'] = 1
#         elif key == 'Smoking Status':
#             if value == 'Occasional':
#                 df['smoking_status_Occasional'] = 1
#             elif value == 'Regular':
#                 df['smoking_status_Regular'] = 1
#         elif key == 'Employment Status':
#             if value == 'Salaried':
#                 df['employment_status_Salaried'] = 1
#             elif value == 'Self-Employed':
#                 df['employment_status_Self-Employed'] = 1
#         elif key == 'Insurance Plan':  # Correct key usage with case sensitivity
#             df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
#         elif key == 'Age':  # Correct key usage with case sensitivity
#             df['age'] = value
#         elif key == 'Number of Dependants':  # Correct key usage with case sensitivity
#             df['number_of_dependants'] = value
#         elif key == 'Income in Lakhs':  # Correct key usage with case sensitivity
#             df['income_lakhs'] = value
#         elif key == "Genetical Risk":
#             df['genetical_risk'] = value
#
#     # Assuming the 'normalized_risk_score' needs to be calculated based on the 'age'
#     df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
#     df = handle_scaling(input_dict['Age'], df)
#
#     return df
#
# def handle_scaling(age, df):
#     # scale age and income_lakhs column
#     if age <= 25:
#         scaler_object = scaler_young
#     else:
#         scaler_object = scaler_rest
#
#     cols_to_scale = scaler_object['cols_to_scale']
#     scaler = scaler_object['scaler']
#
#     df['income_level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
#     df[cols_to_scale] = scaler.transform(df[cols_to_scale])
#
#     df.drop('income_level', axis='columns', inplace=True)
#
#     return df
#
# def predict(input_dict):
#     input_df = preprocess_input(input_dict)
#
#     if input_dict['Age'] <= 25:
#         prediction = model_young.predict(input_df)
#     else:
#         prediction = model_rest.predict(input_df)
#
#     return int(prediction[0])
#
#

# # chatgpt
# import pandas as pd
# import joblib
# import os
#
# # Load models and scalers with correct paths
# model_young_path = os.path.join("artifacts", "model_young.joblib")
# model_rest_path = os.path.join("artifacts", "model_rest.joblib")
# scaler_young_path = os.path.join("artifacts", "scaler_young.joblib")
# scaler_rest_path = os.path.join("artifacts", "scaler_rest.joblib")
#
# try:
#     model_young = joblib.load(model_young_path)
#     model_rest = joblib.load(model_rest_path)
#     scaler_young = joblib.load(scaler_young_path)
#     scaler_rest = joblib.load(scaler_rest_path)
# except FileNotFoundError as e:
#     raise FileNotFoundError(f"Missing model or scaler file: {e}")
#
#
# def calculate_normalized_risk(medical_history):
#     risk_scores = {
#         "diabetes": 6,
#         "heart disease": 8,
#         "high blood pressure": 6,
#         "thyroid": 5,
#         "no disease": 0,
#         "none": 0
#     }
#     diseases = medical_history.lower().split(" & ")
#     total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
#     max_score = 14  # Normalized max risk
#     return total_risk_score / max_score if max_score else 0
#
#
# def preprocess_input(input_dict):
#     expected_columns = [
#         'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
#         'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
#         'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
#         'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
#     ]
#
#     df = pd.DataFrame(0, columns=expected_columns, index=[0])
#     insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
#
#     for key, value in input_dict.items():
#         if key == 'Gender' and value == 'Male':
#             df['gender_Male'] = 1
#         elif key == 'Region':
#             df[f'region_{value}'] = 1
#         elif key == 'Marital Status' and value == 'Unmarried':
#             df['marital_status_Unmarried'] = 1
#         elif key == 'BMI Category':
#             df[f'bmi_category_{value}'] = 1
#         elif key == 'Smoking Status':
#             df[f'smoking_status_{value}'] = 1
#         elif key == 'Employment Status' and value in ['Salaried', 'Self-Employed']:
#             df[f'employment_status_{value}'] = 1
#         elif key == 'Insurance Plan':
#             df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
#         elif key in ['Age', 'Number of Dependants', 'Income in Lakhs', 'Genetical Risk']:
#             df[key.lower().replace(" ", "_")] = value
#
#     df['normalized_risk_score'] = calculate_normalized_risk(input_dict['Medical History'])
#     return handle_scaling(input_dict['Age'], df)
#
#
# def handle_scaling(age, df):
#     scaler_object = scaler_young if age <= 25 else scaler_rest
#     cols_to_scale = scaler_object['cols_to_scale']
#     scaler = scaler_object['scaler']
#
#     df = df.drop(columns=[col for col in ['income_level'] if col in df], errors='ignore')
#     df[cols_to_scale] = scaler.transform(df[cols_to_scale])
#     return df
#
#
# def predict(input_dict):
#     input_df = preprocess_input(input_dict)
#     model = model_young if input_dict['Age'] <= 25 else model_rest
#     return int(model.predict(input_df)[0])
#

# gpt2

import os
import pandas as pd
import joblib

# Ensure paths are absolute and correct for macOS
BASE_DIR = os.path.dirname(__file__)  # Get current file's directory
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

model_young_path = os.path.join(ARTIFACTS_DIR, "model_young.joblib")
model_rest_path = os.path.join(ARTIFACTS_DIR, "model_rest.joblib")
scaler_young_path = os.path.join(ARTIFACTS_DIR, "scaler_young.joblib")
scaler_rest_path = os.path.join(ARTIFACTS_DIR, "scaler_rest.joblib")

# Load models and scalers
model_young = joblib.load(model_young_path)
model_rest = joblib.load(model_rest_path)
scaler_young = joblib.load(scaler_young_path)
scaler_rest = joblib.load(scaler_rest_path)


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

    max_score = 14  # Maximum risk score
    min_score = 0  # Minimum risk score

    # Normalize the score
    return (total_risk_score - min_score) / (max_score - min_score)


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

    # Compute normalized risk score
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'none'))

    # Apply scaling
    df = handle_scaling(input_dict.get('Age', 0), df)

    return df


def handle_scaling(age, df):
    # Select appropriate scaler
    scaler_object = scaler_young if age <= 25 else scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    # Ensure 'income_level' is handled properly
    df['income_level'] = 0
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop('income_level', axis=1, inplace=True)

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)

    model = model_young if input_dict.get('Age', 0) <= 25 else model_rest
    prediction = model.predict(input_df)

    return int(prediction[0])
