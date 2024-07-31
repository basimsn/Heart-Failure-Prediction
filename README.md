Heart Failure Prediction and Risk Stratification
This project involves the analysis and prediction of heart failure events using machine learning models. The primary goal is to classify patients based on their likelihood of experiencing heart failure (denoted as DEATH_EVENT) and identify key risk factors contributing to these outcomes.

Table of Contents
Project Overview
Data Description
Installation
Usage
Models Implemented
Decision Tree
Logistic Regression
Random Forest
Risk Stratification
Feature Importance
Results and Performance
Conclusion
References
Project Overview
This project uses clinical records data to predict the likelihood of heart failure in patients and stratifies them into high-risk and low-risk groups. The dataset contains multiple features representing clinical measurements and the target variable DEATH_EVENT, which indicates whether a patient experienced a heart failure event.

Data Description
The dataset used for this project includes the following features:

age: Age of the patient
anaemia: Decrease in red blood cells or hemoglobin (0: No, 1: Yes)
creatinine_phosphokinase: Level of the CPK enzyme in the blood (mcg/L)
diabetes: If the patient has diabetes (0: No, 1: Yes)
ejection_fraction: Percentage of blood leaving the heart at each contraction
high_blood_pressure: If the patient has hypertension (0: No, 1: Yes)
platelets: Platelet count in the blood (kiloplatelets/mL)
serum_creatinine: Level of serum creatinine in the blood (mg/dL)
serum_sodium: Level of serum sodium in the blood (mEq/L)
sex: Gender of the patient (0: Female, 1: Male)
smoking: If the patient smokes (0: No, 1: Yes)
time: Follow-up period (days)
DEATH_EVENT: Whether the patient died during the follow-up period (0: No, 1: Yes)
Installation
To run the code, you'll need to install the necessary Python libraries. Use the following command to install the dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Clone the repository to your local machine.
Place the dataset (data.csv) in the project directory.
Run the script using Python:
bash
Copy code
python heart_failure_prediction.py
This will execute the data loading, model training, prediction, and evaluation steps.

Models Implemented
Decision Tree
The Decision Tree model is used to classify patients based on their likelihood of experiencing heart failure. The model is trained using the scikit-learn library and evaluated based on accuracy and classification report metrics.

Logistic Regression
Logistic Regression is implemented to classify the binary outcome (DEATH_EVENT). The model is trained using standardized features and evaluated for accuracy, with feature importance analyzed through coefficients.

Random Forest
The Random Forest model was also trained and evaluated but yielded lower accuracy compared to Logistic Regression and Decision Tree models.

Risk Stratification
Risk stratification is performed to categorize patients into high-risk and low-risk groups based on their predicted probabilities of heart failure events. The thresholds for classification can be adjusted as needed.

Feature Importance
Feature importance is analyzed for the Decision Tree and Logistic Regression models. For Decision Trees, feature importance is determined by the impact of features on the tree's splits. For Logistic Regression, coefficients are used to assess the influence of features on the prediction outcome.

Results and Performance
Logistic Regression achieved the highest accuracy of 86%.
Decision Tree yielded an accuracy of 64%.
Random Forest produced an accuracy of 75%.
Key Findings
Logistic Regression provided the fastest training time and highest accuracy, making it the preferred model for this task.
Age, ejection fraction, and serum_creatinine were identified as significant predictors of heart failure.
Conclusion
This project demonstrates the effectiveness of different machine learning models in predicting heart failure and emphasizes the importance of feature selection in achieving accurate predictions. Logistic Regression outperformed other models in both accuracy and training efficiency, making it a suitable choice for this dataset.

References
Heart Failure Clinical Records Dataset
scikit-learn Documentation
UCI Machine Learning Repository
