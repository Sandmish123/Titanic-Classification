# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np

# # Load your trained models
# svm_model = joblib.load('svm.pkl')
# random_forest_model = joblib.load('random_forest.pkl')
# naive_bayes_model = joblib.load('Naive_bayes.pkl')
# knn_model = joblib.load('KNN.pkl')
# decision_tree_model = joblib.load('Decision_tree.pkl')
# logistic_regression_model = joblib.load('Logistic_regression.pkl')

# # Function to preprocess inputs
# def preprocess_input(pclass, sex, age, fare):
#     sex_num = 1 if sex == 'male' else 0
#     features = np.array([pclass, sex_num, age, fare])
#     return features

# # Function to make predictions
# def predict(model, features):
#     prediction = model.predict(features.reshape(1, -1))
#     return prediction[0]

# # Sidebar for user input
# st.sidebar.title('User Input')
# st.sidebar.write('Provide passenger details:')

# age = st.sidebar.slider('Age', 0, 100, 30)
# fare = st.sidebar.slider('Fare', 0.0, 100.0, 30.0)
# pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
# sex = st.sidebar.selectbox('Sex', ['male', 'female'])

# # Sidebar for model selection
# model_choice = st.sidebar.selectbox('Select Model', ['SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Decision Tree', 'Logistic Regression'])

# # Main page
# st.title('Titanic Survival Prediction App')
# st.write('Enter the values on the sidebar to predict survival on the Titanic.')

# if st.sidebar.button('Predict'):
#     features = preprocess_input(pclass, sex, age, fare)

#     if model_choice == 'SVM':
#         prediction = predict(svm_model, features)
#     elif model_choice == 'Random Forest':
#         prediction = predict(random_forest_model, features)
#     elif model_choice == 'Naive Bayes':
#         prediction = predict(naive_bayes_model, features)
#     elif model_choice == 'KNN':
#         prediction = predict(knn_model, features)
#     elif model_choice == 'Decision Tree':
#         prediction = predict(decision_tree_model, features)
#     elif model_choice == 'Logistic Regression':
#         prediction = predict(logistic_regression_model, features)

#     st.subheader('Prediction Result')
#     if prediction == 1:
#         st.success('The model predicts that the passenger would survive.')
#     else:
#         st.error('The model predicts that the passenger would not survive.')

# # User Guide
# st.sidebar.markdown('---')
# st.sidebar.title('User Guide')
# st.sidebar.write("""
# This app predicts whether a passenger would survive the Titanic disaster based on the following features:

# - **Age**: The age of the passenger.
# - **Fare**: The fare paid by the passenger.
# - **Pclass**: The class of the ticket (1, 2, or 3).
# - **Sex**: The gender of the passenger.

# Select a model from the dropdown and click 'Predict' to see the results.
# """)

# # Insights Section
# st.sidebar.title('Historical Insights')
# st.sidebar.write("""
# From historical data, we know:
# - **Higher class passengers (Pclass 1)** had a higher survival rate.
# - **Women and children** had higher survival rates.
# - **Lower fares** were associated with lower survival rates.
# """)







import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load your trained models
svm_model = joblib.load('svm.pkl')
random_forest_model = joblib.load('random_forest.pkl')
naive_bayes_model = joblib.load('Naive_bayes.pkl')
knn_model = joblib.load('KNN.pkl')
decision_tree_model = joblib.load('Decision_tree.pkl')
logistic_regression_model = joblib.load('Logistic_regression.pkl')

# Function to preprocess inputs
def preprocess_input(pclass, sex, age, fare):
    sex_num = 1 if sex == 'male' else 0
    features = np.array([pclass, sex_num, age, fare])
    return features

# Function to make predictions
def predict(model, features):
    prediction = model.predict(features.reshape(1, -1))
    return prediction[0]

# Main page
st.title('🚢 Titanic Survival Prediction App')
st.write('Enter the values on the sidebar to predict survival on the Titanic.')

st.markdown('---')

st.subheader('📝 User Guide')
st.write("""
This app predicts whether a passenger would survive the Titanic disaster based on the following features:

- **Age**: The age of the passenger.
- **Fare**: The fare paid by the passenger.
- **Pclass**: The class of the ticket (1, 2, or 3).
- **Sex**: The gender of the passenger.

Select a model from the dropdown on the sidebar and click 'Predict' to see the results.
""")

st.markdown('---')

st.subheader('📜 Historical Insights')
st.write("""
From historical data, we know:
- **Higher class passengers (Pclass 1)** had a higher survival rate.
- **Women and children** had higher survival rates.
- **Lower fares** were associated with lower survival rates.
""")

st.markdown('---')

# Sidebar for model selection
st.sidebar.title('🔧 Model Selection')
model_choice = st.sidebar.selectbox('Select Model', ['SVM', 'Random Forest', 'Naive Bayes', 'KNN', 'Decision Tree', 'Logistic Regression'])

# Sidebar for user input
st.sidebar.title('User Input')
st.sidebar.write('Provide passenger details:')

age = st.sidebar.slider('Age', 0, 100, 30)
fare = st.sidebar.slider('Fare', 0.0, 100.0, 30.0)
pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])

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

    st.subheader('🔍 Prediction Result')
    if prediction == 1:
        st.success('🟢 The model predicts that the passenger would survive.')
    else:
        st.error('🔴 The model predicts that the passenger would not survive.')

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Sandeep Mishra](https://your-linkedin-profile)")


# # Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("Developed by [Sandeep Mishra](https://your-linkedin-profile)")
