# Alzheimer Disease

Author : LoveCrush <br>
Time : Within a week <br>
Collaborator : Than. Ng <br>
Goal : Review tools and models for ML <br>
Dataset : Alzheimer Disease <br>

You can follow this link to get fully intuition about this dataset  <a href="https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data">Alzheimer Disease - Kaggle Dataset</a> 


## Motivation

This project's considering as a second project for my uni learning progress. 'Cuz I'm not have any project doing on ML so far. This project was created for that purpose. Just a small project but I'm gonna try my best to apply all of my knowledge into this. So let's briefly describe this dataset for real quick.

## Dataset

Features (cols) : 35 <br>
Samples  (rows) : 2149


| S.No | Name                      | Not Null      |  Dtype                   |  Description  | 
| ---- | ------------------------  | ------------- | ------------------------ | --------      |
|      | **Demographic Detail**    |
| 0    | Age                       | 2149 non-null |  int64   -> numerical    |  Age of patient ranges from 60 -> 90 years   |
| 1    | Gender                    | 2149 non-null |  int64   -> categorical  |  Gender of the patients, O represents Male and vice versa    |
| 2    | Ethnicity                 | 2149 non-null |  int64   -> categorical  |  The ethnicity of the patients, already encoded (irrelevant in detail)       |
| 3    | EducationLevel            | 2149 non-null |  int64   -> categorical  |  The educational level of patient, already encoded (irrelevant in detail)        |
|      | **Lifestyle Factor**      |
| 4    | BMI                       | 2149 non-null |  float64 -> numerical    |  Body Mass Index of the patients, ranging from 15 -> 40        |
| 5    | Smoking                   | 2149 non-null |  int64   -> categorical  |  Smoking status, where 0 indicates No and 1 indicates Yes        |
| 6    | AlcoholConsumption        | 2149 non-null |  float64 -> numerical    |  Weekly alcohol consumption in units, ranging from 0 -> 20        |
| 7    | PhysicalActivity          | 2149 non-null |  float64 -> numerical    |  Weekly physical activity in hours, ranging from 0 -> ten        |
| 8    | DietQuality               | 2149 non-null |  float64 -> numerical    |  Diet quality score, ranging from 0 -> 10         |
| 9    | SleepQuality              | 2149 non-null |  float64 -> numerical    |  Sleep quality score, ranging from 4 -> 10        |
|      | **Mediacal Hostory**      | 
| 10   | FamilyHistoryAlzheimers   | 2149 non-null |  int64   -> categorical  |  Family history of Alzheimer's Disease, 0 indicated No and 1 indicated Yes         |
| 11   | CardiovascularDisease     | 2149 non-null |  int64   -> categorical  |  Presence of Cardiovascular Disease, 0 indicates No, 1 indicates Yes         |
| 12   | Diabetes                  | 2149 non-null |  int64   -> categorical  |  Presence of diabetes, 0 indicates No and 1 indicates Yes        |
| 13   | Depression                | 2149 non-null |  int64   -> categorical  |  Presence of Depression, where 0 indicates No and 1 indicates Yes        |
| 14   | HeadInjury                | 2149 non-null |  int64   -> categorical  |  History of head injury, 0 indicates No and 1 indicates Yes        |
| 15   | Hypertension              | 2149 non-null |  int64   -> categorical  |  Presence of Hypertension, 0 indicates No and 1 indicates Yes        |
|      | **Clinical Measurements** |
| 16   | SystolicBP                | 2149 non-null |  int64   -> numerical    |  Systolic blood pressure, ranging from 90 -> 180 mmHg        |
| 17   | DiastolicBP               | 2149 non-null |  int64   -> numerical    |  Diastolic blood pressure, ranging from 60 to 120 mmHg        |
| 18   | CholesterolTotal          | 2149 non-null |  float64 -> numerical    |  Total Cholesterol levels, ranging from 150 -> 300 mm/dL        |
| 19   | CholesterolLDL            | 2149 non-null |  float64 -> numerical    |  Low-density lipoprotein cholesterol levels, ranging from 50 -> 200 mg/dL          |
| 20   | CholesterolHDL            | 2149 non-null |  float64 -> numerical    |  High-density lipoprotein cholesterol levels, ranging from 20 -> 100 mm/dL        |
| 21   | CholesterolTriglycerides  | 2149 non-null |  float64 -> numerical    |  Triglycerides levels, ranging from 50 -> 400 mg/dL.        |
|      | **Cognitive and Functional Assessments**  |
| 22   | MMSE                      | 2149 non-null |  float64 -> numerical    |  Mini-Mental State Examination score, ranging from 0 -> 30, lower scores indicate cognitive impairment        |
| 23   | FunctionalAssessment      | 2149 non-null |  float64 -> numerical    |  Functional assessment score, ranging from 0 -> 10, lower scores indicate greater impairment        |
| 24   | MemoryComplaints          | 2149 non-null |  int64   -> categorical  |  Presence of Memory complaints, where 0 indicates No and 1 indicates Yes        |
| 25   | BehavioralProblems        | 2149 non-null |  int64   -> categorical  |  Presence of Behavioral problem, where 0 indicates No and 1 indicate Yes        |
| 26   | ADL                       | 2149 non-null |  float64 -> numerical    |  Activities of Daily Living score, ranging from 0 -> 10, lower scores indicate greater impairment        |
|      | **Symptoms**              |
| 27   | Confusion                 | 2149 non-null |  int64   -> categorical  |  Presence of confusion, 0 indicated No and 1 indicated Yes        |
| 28   | Disorientation            | 2149 non-null |  int64   -> categorical  |  Presence of disorientation, 0 indicates No and 1 indicates Yes       |
| 29   | PersonalityChanges        | 2149 non-null |  int64   -> categorical  |  Presence of personality changes, 0 indicates No and 1 indicates Yes        |
| 30   | DifficultyCompletingTasks | 2149 non-null |  int64   -> categorical  |  Presence of difficulty completing tasks, where 0 indicates No and 1 indicates Yes         |
| 31   | Forgetfulness             | 2149 non-null |  int64   -> categorical  |  Presence of forgetfulness, where 0 indicates No and 1 indicates Yes        |
|      | **Diagnosis Information** |
| 32   | Diagnosis                 | 2149 non-null |  int64   -> categorical  |  Diagnosis status for Alzheimer's Disease, where 0 indicates No and 1 indicates Yes         |
|      | **Confidential Information**              |
| 33   | DoctorInCharge            | 2149 non-null |  object  -> categorical  | (irrelevant) The confidential information about the doctor in charge, already encoded as "XXXConfid for all patients         |
| 34   | PatientID                 | 2149 non-null |  int64   -> numerical    | (irrelevant) A unique identifier assigned to each patient   |





























