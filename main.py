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






def predict(features):
    with open("saved_models/GaussianNB.pickle", 'rb') as file:
        model = pickle.load(file)

    print(model[0].predict(np.array(features)))


def GUI():
    """"""  
    st.title("Alzheimer's Disease Diagnosis")          
    st.text("Write some descriptions here! ")          
    
    with st.sidebar:                                    
        st.title("Sidebar Title")
        st.button("SIdebar button")
        st.button("st button")
        st.radio("Radio", ["love", "crush"])
        st.header("Lovecrush")
        st.radio("lovecrush", ["love"])
        st.multiselect("Love",["Love", "crush"])
        with st.expander("Sidebar expander"):
            st.button("lovecrush")

    tabs = st.tabs(["Input", "File", "Result", "Report"])      

    with tabs[0]:
        st.write("Do something here!,  tab input")

        with st.form("input-form"):

            st.subheader("Patient Information")
            patient_info = [st.text_input("Patient Id:", value=121 , disabled=True),
                            st.text_input("Your name", value="LoveCrush")]

            st.subheader("Demographic Details")
            demo_detail = [st.text_input("Age", value="60"),
                            st.radio("Gender", options=["Male", "Female"]),
                            st.radio("Ethnicity", options=["Caucasian", "African", "Asian", "Other"]),
                            st.radio("Education Level", options=["None", "High school", "Bachelor's", "Higher"])]

            st.subheader("Lifestyle Factor")
            life_fact = [st.slider("BMI", min_value=15, max_value=40, value=30),
                        st.radio("Smoking", options=["Yes", "No"]),
                        st.slider("Alcohol consumption", min_value=0, max_value=20),
                        st.slider("Physical activity", min_value=0, max_value=10),
                        st.slider("Diet Quality", min_value=0, max_value=10),
                        st.slider("Sleep Quality", min_value=4, max_value=10)]

            st.subheader("Medical history")
            med_his = [st.radio("FamilyHistoryAlzheimers", options=["Yes", "No"]),
                        st.radio("Diabetes", options=["Yes", "No"]),
                        st.radio("Depression", options=["Yes", "No"]),
                        st.radio("HeadInjury", options=["Yes", "No"]),
                        st.radio("Hypertension", options=["Yes", "No"])]

            st.subheader("Clinical Measurements")
            clinic_mea = [st.slider("SystolicBP",90,180, 100),
                            st.slider("DiastolicBP",60,120, 100),
                            st.slider("CholesterolTotal",150,300, 100),
                            st.slider("CholesterolLDL",50,200, 100),
                            st.slider("CholesterolHDL",20,100, 100),
                            st.slider("CholesterolTriglycerides",50,400, 100)]

            st.subheader("Cognitive and Functional Assessments")
            cog_fun = [st.slider("MMSE", 0,30,15),
                        st.slider("FunctionalAssessment", 0,10, 5),
                        st.radio("MemoryComplaint", options=["Yes", "No"]),
                        st.radio("BehavioralProblems", options=["Yes", "No"]),
                        st.slider("ADL", 0,10)]

            st.subheader("Symptoms")
            symp = [st.selectbox("Confusion", options=["Yes", "No"]),
                    st.selectbox("Disorientation", options=["Yes", "No"]),
                    st.selectbox("PersonalityChanges", options=["Yes", "No"]),
                    st.selectbox("DifficultyCompletingTasks", options=["Yes", "No"]),
                    st.selectbox("Forgetfulness", options=["Yes", "No"])]
            
            st.subheader("Confidential Information")
            conf = [st.text_input("DoctorInCharge",value="VuQuangPhuc", disabled=True),
                    st.date_input("Date", value=datetime.now()),
                    st.time_input("Time", value=datetime.now()),
                    st.text_area("Your information")]


            if st.form_submit_button("Diagnose", type="primary"):        # Summit event
                st.write("Submitted ", patient_info,
                                        demo_detail,
                                        life_fact,
                                        med_his,
                                        clinic_mea,
                                        cog_fun,
                                        symp,
                                        conf)
                
                features = []
                features.append(patient_info[0])
                features.extend(demo_detail)
                features.extend(life_fact)
                features.extend(med_his)
                features.extend(clinic_mea)
                features.extend(cog_fun)
                features.extend(symp)
                features.append(conf[0])
                
                print(features, f"len {len(features)}")
                                        
                                        
                prediction = predict(features)                        
                                        
                            

                with st.spinner("Loading..."):
                    time.sleep(5)
                st.success("Done!")


            

    with tabs[1]:
        st.write("Do something here!,  tab File")
        st.write(st.file_uploader("Choose your files:", type=["PNG", "JPG"]))

        df = pd.read_csv("data/alzheimer_disease_data_custom.csv")
        st.dataframe(df)
        




    
            
    

    

    

def main():
    """"""
    # Initialize GUI

    GUI()




if __name__ == "__main__":
    main()
    

    




