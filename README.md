# Titanic Classification

This project is aimed at predicting whether a passenger aboard the Titanic would survive or not based on various features such as socio-economic status, age, gender, and more. It leverages machine learning techniques to build a robust classification model.

## Problem Statement
The objective is to build a system that predicts whether a passenger will survive the Titanic disaster. Key factors influencing survival include socio-economic status, age, gender, and other attributes.

## Solution Approach
1. **Exploratory Data Analysis (EDA)**:
   - Performed EDA on the Titanic dataset to understand the relationships between various features and the target variable (`Survived`).
   - Analyzed and visualized the impact of features like age, fare, class, and gender on survival rates.

2. **Data Preprocessing**:
   - Converted categorical data into numerical values using label encoding.
   - Selected relevant columns for model building based on observations, including `Age`, `Fare`, `Pclass`, and `Sex`.

3. **Model Training**:
   - Applied six different machine learning algorithms:
     - Logistic Regression
     - Decision Tree
     - K-Nearest Neighbors (KNN)
     - Naive Bayes
     - Support Vector Machine (SVM)
     - Random Forest
   - Tuned hyperparameters and evaluated each model's performance.

4. **Best Model**:
   - Achieved the highest accuracy of **81%** using the Decision Tree model with a maximum depth of 5.

## Project Files
- **app.py**: Flask application to serve the model and interact with users.
- **Titanic_Classification_BY_SANDEEP_MISHRA_TASK_1.ipynb**: Jupyter notebook for EDA, preprocessing, and model training.
- **train.csv**: Training dataset used for model development.
- **test.csv**: Test dataset for evaluation.
- **gender_submission.csv**: Sample submission file.
- **Saved Models**:
  - `Logistic_regression.pkl`
  - `Decision_tree.pkl`
  - `KNN.pkl`
  - `Naive_bayes.pkl`
  - `SVM.pkl`
  - `Random_forest.pkl`
- **requirements.txt**: List of dependencies required to run the project.

## Steps to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Sandmish123/Titanic-Classification.git
   cd Titanic-Classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Access the application at `http://localhost:5000` in your browser.

## Key Insights
- **Impact of Gender**: Females had a significantly higher survival rate than males.
- **Socio-Economic Status**: Passengers in higher classes (Pclass 1) were more likely to survive.
- **Age**: Younger passengers had higher survival rates.
- **Fare**: Higher ticket prices correlated with a better chance of survival.

## Tools and Libraries
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy (Data Manipulation)
  - Matplotlib, Seaborn (Data Visualization)
  - Scikit-learn (Model Building)
  - Flask (Web Framework)

## Future Enhancements
1. Improve feature engineering to include interactions between variables.
2. Implement advanced algorithms like XGBoost and Neural Networks for better accuracy.
3. Build a user-friendly front-end for easier interaction.

## Conclusion
This project demonstrates a complete machine learning pipeline, from data preprocessing and EDA to model training and deployment. It provides a practical approach to solving classification problems and offers insights into the factors influencing survival on the Titanic.

Feel free to fork this repository and contribute to its development!

