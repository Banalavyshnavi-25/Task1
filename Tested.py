import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import streamlit as st
import sweetviz as sv
import dtale

# Title and Description
st.title("Titanic Survival Prediction")
st.write("This app performs EDA on the Titanic dataset and trains a RandomForestClassifier to predict survival.")

# Uploading the Dataset
uploaded_file = st.file_uploader(r"C:\Users\91739\OneDrive\Documents\Python Scripts\tested.csv", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")

    # Display the dataset
    st.subheader("Dataset Preview")
    st.write(data.head())

    # EDA Information
    if st.checkbox("Show Dataset Info"):
        st.write(data.info())

    if st.checkbox("Show Dataset Description"):
        st.write(data.describe())

    # Check for Null Values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Removing unnecessary columns
    data = data.drop(columns=['Name', 'Ticket', 'Fare', 'Cabin', 'Embarked'], errors='ignore')

    # Visualizations
    st.subheader("Survivors by Class")
    pclass_sur = data[['Pclass', 'Survived']].groupby('Pclass').sum()
    st.bar_chart(pclass_sur)

    st.subheader("Survivors by Gender")
    sex_sur = data[['Sex', 'Survived']].groupby('Sex').sum()
    st.bar_chart(sex_sur)

    # Prepare Data for Training
    x = data.drop(columns='Survived', errors='ignore')
    y = data['Survived']

    # Encoding Categorical Variables
    label = LabelEncoder()
    x['Sex'] = label.fit_transform(x['Sex'])

    # Splitting the Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Training the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Evaluating the Model
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    # AutoEDA with Sweetviz
    if st.checkbox("Generate Sweetviz Report"):
        report = sv.analyze(data)
        report.show_html('report.html')
        st.markdown("Sweetviz report generated! Check 'report.html' in your directory.")
    
    # D-Tale
    if st.checkbox("Open D-Tale for Data Exploration"):
        with dtale.show(data) as d:
            d.open_browser()