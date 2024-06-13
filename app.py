import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Path to the CSV file
csv_file_path = 'titanic_data.csv'

# Load data with error handling
if os.path.exists(csv_file_path):
    data = pd.read_csv(csv_file_path)
else:
    st.error(f"File '{csv_file_path}' not found. Please ensure the file is in the correct directory.")

# Load your trained models
svm_model = joblib.load('svm.pkl')
random_forest_model = joblib.load('random_forest.pkl')
naive_bayes_model = joblib.load('Naive_bayes.pkl')
knn_model = joblib.load('KNN.pkl')
decision_tree_model = joblib.load('Decision_tree.pkl')
logistic_regression_model = joblib.load('Logistic_regression.pkl')

# Function to preprocess inputs
def preprocess_input(pclass, sex, age, fare):
    # Convert sex to numerical
    sex_num = 1 if sex == 'male' else 0
    # Create a numpy array with the processed features
    features = np.array([pclass, sex_num, age, fare])
    return features

# Function to make predictions
def predict(model, features):
    prediction = model.predict(features.reshape(1, -1))
    return prediction[0]  # Return the predicted class (0 or 1)

def main():
    st.title('Titanic Survival Prediction App')
    st.write('Enter the values to predict survival on the Titanic')

    # User inputs for prediction
    age = st.slider('Age', 0, 100, 30)
    fare = st.slider('Fare', 0.0, 100.0, 30.0)
    pclass = st.selectbox('Pclass', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])

    # Model selection
    model_choice = st.selectbox('Select Model', ['SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Decision Tree', 'Logistic Regression'])

    # Prediction button
    if st.button('Predict'):
        # Preprocess the input features
        features = preprocess_input(pclass, sex, age, fare)

        # Make prediction based on the selected model
        if model_choice == 'SVM':
            prediction = predict(svm_model, features)
        elif model_choice == 'Random Forest':
            prediction = predict(random_forest_model, features)
        elif model_choice == 'Naive Bayes':
            prediction = predict(naive_bayes_model, features)
        elif model_choice == 'KNN':
            prediction = predict(knn_model, features)
        elif model_choice == 'Decision Tree':
            prediction = predict(decision_tree_model, features)
        elif model_choice == 'Logistic Regression':
            prediction = predict(logistic_regression_model, features)

        # Display prediction result
        if prediction == 1:
            st.success('The model predicts that the passenger would survive.')
        else:
            st.error('The model predicts that the passenger would not survive.')

if __name__ == '__main__':
    main()
