import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your trained models
svm_model = joblib.load('svm.pkl')
random_forest_model = joblib.load('random_forest.pkl')
naive_bayes_model = joblib.load('Naive_bayes.pkl')
knn_model = joblib.load('KNN.pkl')
decision_tree_model = joblib.load('Decision_tree.pkl')
logistic_regression_model = joblib.load('Logistic_regression.pkl')

# Load dataset for EDA
data = pd.read_csv('titanic_data.csv')

# Function to preprocess inputs
def preprocess_input(pclass, sex, age, fare):
    sex_num = 1 if sex == 'male' else 0
    features = np.array([pclass, sex_num, age, fare])
    return features

# Function to make predictions
def predict(model, features):
    prediction = model.predict(features.reshape(1, -1))
    return prediction[0]

# Sidebar for user input
st.sidebar.title('User Input')
st.sidebar.write('Provide passenger details:')

age = st.sidebar.slider('Age', 0, 100, 30)
fare = st.sidebar.slider('Fare', 0.0, 100.0, 30.0)
pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])

# Sidebar for model selection
model_choice = st.sidebar.selectbox('Select Model', ['SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Decision Tree', 'Logistic Regression'])

# Main page
st.title('Titanic Survival Prediction App')
st.write('Enter the values on the sidebar to predict survival on the Titanic.')

if st.sidebar.button('Predict'):
    features = preprocess_input(pclass, sex, age, fare)

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

    st.subheader('Prediction Result')
    if prediction == 1:
        st.success('The model predicts that the passenger would survive.')
    else:
        st.error('The model predicts that the passenger would not survive.')

# EDA Section
st.sidebar.markdown('---')
st.sidebar.title('Explore Dataset')
if st.sidebar.checkbox('Show Dataset'):
    st.subheader('Titanic Dataset')
    st.write(data)

# Display correlation heatmap
if st.sidebar.checkbox('Show Correlation Heatmap'):
    st.subheader('Correlation Heatmap')
    corr = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, ax=ax, annot=True, cmap='coolwarm')
    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Your Name](https://your-linkedin-profile)")
