
# Customer Churn Prediction Using Machine Learning

# 1. Project Overview

An end-to-end machine learning project that predicts whether a telecom customer is likely to leave a service provider.

The objective is to identify potential customer churn using historical customer data, helping businesses improve customer retention strategies.

# 2. Dataset

 Dataset: IBM Telco Customer Churn
 Total observations: 7,043
 Original variables: 21
 Input features: 19
 Target variable: Churn (Yes/No)

# 3. Machine Learning Models

Implemented and compared three classification algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

Model performance was evaluated using:

 Accuracy
 Precision
 Recall
 F1-score
 ROC-AUC

# 4. Project Structure

text
customer-churn-ml-project/
│
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed_dataset.csv
│
├── models/
│   └── churn_prediction_model.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── app.py
├── project report.pdf
├── README.md
├── requirements.txt
└── .gitignore


# 5. Technologies Used

 Python
 Pandas and NumPy
 Scikit-learn
 Matplotlib and Seaborn
 Jupyter Notebook
 Streamlit
 Joblib

# 6. Installation and Usage

Clone the repository:


git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-churn-ml-project


Create a virtual environment:


python -m venv .venv


Activate it on Windows:


.venv\Scripts\activate


Install dependencies:


pip install -r requirements.txt


Run the Streamlit application:


python -m streamlit run app.py


Open the local URL displayed in the terminal.

# 7. Model and Application

The trained machine learning pipeline is saved as:

`models/churn_prediction_model.pkl`

The Streamlit application loads the saved model and predicts customer churn based on user-provided customer information.

# 8. Project Deliverables

 Data exploration and model training notebook
 Trained machine learning model
 Original and processed datasets
 Streamlit prediction application



