"""
Created on Dec, 23, 2024 , 14:00

@author: lovecrush
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import time
import sklearn
import asyncio
import os

def predict(features, model_name):
    try:
        with open(f"saved_models/{model_name}.pickle", 'rb') as file:
            model = pickle.load(file)
        
        # Convert features to appropriate format
        features_array = np.array(features).reshape(1, -1)
        
        # Print shape for debugging
        print(f"Features shape: {features_array.shape}")
        
        # Make prediction
        prediction = model[0].predict(features_array)
        
        # Try to get probability if available, otherwise return None
        try:
            probability = model[0].predict_proba(features_array)
            return prediction[0], probability[0]
        except AttributeError:
            # If predict_proba is not available, return prediction with None probability
            return prediction[0], None
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        # Print more detailed error information
        print(f"Detailed error: {str(e)}")
        print(f"Features shape: {np.array(features).shape}")
        return None, None

def process_form_data(patient_info, demo_detail, life_fact, med_his, clinic_mea, cog_fun, symp, conf):
    # Convert categorical variables to numerical
    gender_map = {"Male": 1, "Female": 0}
    yes_no_map = {"Yes": 1, "No": 0}
    ethnicity_map = {"Caucasian": 0, "African": 1, "Asian": 2, "Other": 3}
    education_map = {"None": 0, "High school": 1, "Bachelor's": 2, "Higher": 3}
    
    processed_features = []
    
    # Process patient info (1 feature)
    processed_features.append(int(patient_info[0]))  # Patient ID
    
    # Process demographic details (4 features)
    processed_features.extend([
        int(demo_detail[0]),  # Age
        gender_map[demo_detail[1]],  # Gender
        ethnicity_map[demo_detail[2]],  # Ethnicity
        education_map[demo_detail[3]]   # Education Level
    ])
    
    # Process lifestyle factors (6 features)
    processed_features.extend([
        float(life_fact[0]),  # BMI
        yes_no_map[life_fact[1]],  # Smoking
        float(life_fact[2]),  # Alcohol consumption
        float(life_fact[3]),  # Physical activity
        float(life_fact[4]),  # Diet Quality
        float(life_fact[5])   # Sleep Quality
    ])
    
    # Process medical history (5 features)
    processed_features.extend([
        yes_no_map[med_his[0]],  # FamilyHistoryAlzheimers
        yes_no_map[med_his[1]],  # Diabetes
        yes_no_map[med_his[2]],  # Depression
        yes_no_map[med_his[3]],  # HeadInjury
        yes_no_map[med_his[4]]   # Hypertension
    ])
    
    # Process clinical measurements (6 features)
    processed_features.extend([
        float(clinic_mea[0]),  # SystolicBP
        float(clinic_mea[1]),  # DiastolicBP
        float(clinic_mea[2]),  # CholesterolTotal
        float(clinic_mea[3]),  # CholesterolLDL
        float(clinic_mea[4]),  # CholesterolHDL
        float(clinic_mea[5])   # CholesterolTriglycerides
    ])
    
    # Process cognitive and functional assessments (5 features)
    processed_features.extend([
        float(cog_fun[0]),  # MMSE
        float(cog_fun[1]),  # FunctionalAssessment
        yes_no_map[cog_fun[2]],  # MemoryComplaint
        yes_no_map[cog_fun[3]],  # BehavioralProblems
        float(cog_fun[4])   # ADL
    ])
    
    # Process symptoms (5 features)
    processed_features.extend([
        yes_no_map[symp[0]],  # Confusion
        yes_no_map[symp[1]],  # Disorientation
        yes_no_map[symp[2]],  # PersonalityChanges
        yes_no_map[symp[3]],  # DifficultyCompletingTasks
        yes_no_map[symp[4]]   # Forgetfulness
    ])
    
    # Add doctor information (1 feature)
    processed_features.append(1 if conf[0] == "VuQuangPhuc" else 0)  # DoctorInCharge
    
    # Add date and time features (2 features)
    date_time = conf[1]  # Date
    processed_features.extend([
        date_time.month,  # Month as a feature
        date_time.year    # Year as a feature
    ])
    
    # Print feature count for debugging
    print(f"Total features: {len(processed_features)}")
    print("Features:", processed_features)
    
    return processed_features

def get_available_models():
    """Get list of available models from saved_models directory"""
    # Get all .pickle files from saved_models directory
    model_files = [f for f in os.listdir("saved_models") if f.endswith('.pickle') and os.path.getsize(os.path.join("saved_models", f)) > 0]
    
    # Create list of available models (just the names without .pickle extension)
    available_models = [f.replace('.pickle', '') for f in model_files]
    
    return available_models

def GUI():
    st.set_page_config(page_title="Alzheimer's Disease Diagnosis", layout="wide")
    
    # Create two columns for the main layout
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.title("Settings")
        st.markdown("---")
        
        # Model Selection
        st.subheader("Select Model")
        available_models = get_available_models()
        
        if not available_models:
            st.error("No valid models found in the saved_models directory!")
            return
            
        selected_model = st.selectbox(
            "Choose a model for prediction:",
            options=available_models,
            index=0
        )
        
        st.markdown("---")
        st.subheader("Model Information")
        st.info(f"Currently using: {selected_model}")
        
        st.markdown("---")
        st.subheader("Quick Links")
        st.button("Home")
        st.button("About")
        st.markdown("---")
        st.subheader("Contact")
        st.markdown("For support, contact: support@example.com")
    
    with right_col:
        st.title("Alzheimer's Disease Diagnosis")          
        st.markdown("""
        This application helps in diagnosing Alzheimer's Disease based on various patient parameters.
        Please fill in all the required information below for an accurate assessment.
        """)          

        tabs = st.tabs(["Input", "File", "Result", "Report"])      

        with tabs[0]:
            st.write("Please fill in the patient information below")

            with st.form("input-form"):
                # Create three columns for the form
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Personal Information Section
                    st.markdown("### 👤 Personal Information")
                    with st.container():
                        st.markdown("#### Basic Details")
                        patient_info = [
                            st.text_input("Patient ID", value=121, disabled=True),
                            st.text_input("Patient Name", value="LoveCrush")
                        ]
                        
                        st.markdown("#### Demographic Information")
                        demo_detail = [
                            st.number_input("Age", min_value=0, max_value=120, value=60, step=1),
                            st.radio("Gender", options=["Male", "Female"], horizontal=True),
                            st.selectbox("Ethnicity", options=["Caucasian", "African", "Asian", "Other"]),
                            st.selectbox("Education Level", options=["None", "High school", "Bachelor's", "Higher"])
                        ]
                    
                    # Lifestyle Section
                    st.markdown("### 🏃 Lifestyle Factors")
                    with st.container():
                        life_fact = [
                            st.slider("BMI", min_value=15.0, max_value=40.0, value=30.0, step=0.1,
                                    help="Body Mass Index"),
                            st.radio("Smoking Status", options=["Yes", "No"], horizontal=True),
                            st.slider("Alcohol Consumption (units/week)", min_value=0, max_value=20, value=0,
                                    help="Average weekly alcohol consumption"),
                            st.slider("Physical Activity (hours/week)", min_value=0, max_value=10, value=3,
                                    help="Weekly physical activity duration"),
                            st.slider("Diet Quality (1-10)", min_value=1, max_value=10, value=5,
                                    help="Overall diet quality score"),
                            st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=5,
                                    help="Average sleep quality score")
                        ]

                with col2:
                    # Medical History Section
                    st.markdown("### 🏥 Medical History")
                    with st.container():
                        st.markdown("#### Family & Medical Conditions")
                        med_his = [
                            st.radio("Family History of Alzheimer's", options=["Yes", "No"], horizontal=True),
                            st.radio("Diabetes", options=["Yes", "No"], horizontal=True),
                            st.radio("Depression", options=["Yes", "No"], horizontal=True),
                            st.radio("History of Head Injury", options=["Yes", "No"], horizontal=True),
                            st.radio("Hypertension", options=["Yes", "No"], horizontal=True)
                        ]
                    
                    # Clinical Measurements Section
                    st.markdown("### 📊 Clinical Measurements")
                    with st.container():
                        st.markdown("#### Blood Pressure")
                        clinic_mea = [
                            st.slider("Systolic BP (mmHg)", 90, 180, 120,
                                    help="Systolic blood pressure measurement"),
                            st.slider("Diastolic BP (mmHg)", 60, 120, 80,
                                    help="Diastolic blood pressure measurement")
                        ]
                        
                        st.markdown("#### Cholesterol Levels")
                        clinic_mea.extend([
                            st.slider("Total Cholesterol (mg/dL)", 150, 300, 200,
                                    help="Total cholesterol level"),
                            st.slider("LDL Cholesterol (mg/dL)", 50, 200, 100,
                                    help="Low-density lipoprotein level"),
                            st.slider("HDL Cholesterol (mg/dL)", 20, 100, 50,
                                    help="High-density lipoprotein level"),
                            st.slider("Triglycerides (mg/dL)", 50, 400, 150,
                                    help="Triglyceride level")
                        ])

                with col3:
                    # Cognitive Assessment Section
                    st.markdown("### 🧠 Cognitive Assessment")
                    with st.container():
                        st.markdown("#### Test Scores")
                        cog_fun = [
                            st.slider("MMSE Score (0-30)", 0, 30, 15,
                                    help="Mini-Mental State Examination score"),
                            st.slider("Functional Assessment (0-10)", 0, 10, 5,
                                    help="Daily functional ability score"),
                            st.radio("Memory Complaints", options=["Yes", "No"], horizontal=True),
                            st.radio("Behavioral Problems", options=["Yes", "No"], horizontal=True),
                            st.slider("ADL Score (0-10)", 0, 10, 5,
                                    help="Activities of Daily Living score")
                        ]
                    
                    # Symptoms Section
                    st.markdown("### 🤒 Symptoms")
                    with st.container():
                        symp = [
                            st.selectbox("Confusion", options=["Yes", "No"]),
                            st.selectbox("Disorientation", options=["Yes", "No"]),
                            st.selectbox("Personality Changes", options=["Yes", "No"]),
                            st.selectbox("Difficulty Completing Tasks", options=["Yes", "No"]),
                            st.selectbox("Forgetfulness", options=["Yes", "No"])
                        ]
                    
                    # Confidential Information Section
                    st.markdown("### 🔒 Confidential Information")
                    with st.container():
                        conf = [
                            st.text_input("Doctor in Charge", value="VuQuangPhuc", disabled=True),
                            st.date_input("Date", value=datetime.now()),
                            st.time_input("Time", value=datetime.now()),
                            st.text_area("Additional Notes", height=100)
                        ]

                # Submit Button
                st.markdown("---")
                submit_col1, submit_col2, submit_col3 = st.columns([1,2,1])
                with submit_col2:
                    submitted = st.form_submit_button("Run Diagnosis", type="primary", use_container_width=True)

        # Display results outside the form
        if submitted:
            # Process the form data
            features = process_form_data(
                patient_info, demo_detail, life_fact, med_his,
                clinic_mea, cog_fun, symp, conf
            )
            
            with st.spinner("Analyzing patient data..."):
                prediction, probability = predict(features, selected_model)
                
                if prediction is not None:
                    st.success("Analysis Complete!")
                    
                    # Create a container for results
                    results_container = st.container()
                    with results_container:
                        # Display results in a more organized way
                        st.markdown("### 📊 Diagnosis Results")
                        
                        # Create a two-column layout for results
                        result_col1, result_col2 = st.columns(2)
                        
                        with result_col1:
                            st.markdown("#### Prediction")
                            result = "Positive" if prediction == 1 else "Negative"
                            st.markdown(f"**Diagnosis:** {result}")
                            
                        with result_col2:
                            st.markdown("#### Model Information")
                            if probability is not None:
                                st.markdown(f"**Confidence:** {probability[1]:.2%}")
                            else:
                                st.info("Confidence score not available for this model")
                        
                        # Store the results in session state for other tabs
                        st.session_state['diagnosis_result'] = {
                            'prediction': prediction,
                            'probability': probability,
                            'timestamp': datetime.now(),
                            'patient_id': patient_info[0],
                            'model_used': selected_model
                        }

        with tabs[1]:
            st.write("Upload patient files or view existing data")
            
            uploaded_file = st.file_uploader("Choose your files:", type=["PNG", "JPG", "PDF"])
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                
            st.subheader("Existing Patient Data")
            try:
                df = pd.read_csv("data/alzheimer_disease_data_custom.csv")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

        with tabs[2]:
            st.write("View detailed diagnosis results")
            if 'diagnosis_result' in st.session_state:
                result = st.session_state['diagnosis_result']
                st.subheader(f"Diagnosis for Patient ID: {result['patient_id']}")
                st.write(f"Date: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"Model Used: {selected_model}")
                st.write(f"Prediction: {'Positive' if result['prediction'] == 1 else 'Negative'}")
                if result['probability'] is not None:
                    st.write(f"Confidence: {result['probability'][1]:.2%}")
                else:
                    st.info("Confidence score not available for this model")
            else:
                st.info("No diagnosis results available. Please complete the diagnosis form first.")

        with tabs[3]:
            st.write("Generate detailed reports")
            if 'diagnosis_result' in st.session_state:
                result = st.session_state['diagnosis_result']
                report_data = [
                    f"Patient ID: {result['patient_id']}",
                    f"Date: {result['timestamp']}",
                    f"Model Used: {selected_model}",
                    f"Diagnosis: {'Positive' if result['prediction'] == 1 else 'Negative'}"
                ]
                
                if result['probability'] is not None:
                    report_data.append(f"Confidence: {result['probability'][1]:.2%}")
                else:
                    report_data.append("Confidence: Not available for this model")
                
                st.download_button(
                    label="Download Report",
                    data="\n".join(report_data),
                    file_name="diagnosis_report.txt",
                    mime="text/plain"
                )
            else:
                st.info("No diagnosis results available to generate report.")

def main():
    GUI()

if __name__ == "__main__":
    main()
    

    




