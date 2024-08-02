import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import streamlit as st

# Load the dataset
data = pd.read_csv('/content/Churn_Modelling.csv')

# Define features and target
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

# Define a function to predict churn from user input
def predict_churn_from_input():
    st.title("Customer Churn Prediction")
    
    # Get user input for each feature using Streamlit's widgets
    credit_score = st.number_input("Enter the customer's credit score (300-850):", min_value=300, max_value=850, step=1)
    geography = st.selectbox("Enter the customer's geography:", ['France', 'Germany', 'Spain'])
    gender = st.selectbox("Enter the customer's gender:", ['Male', 'Female'])
    age = st.number_input("Enter the customer's age (18-100):", min_value=18, max_value=100, step=1)
    tenure = st.number_input("Enter the customer's tenure (in years, 0-10):", min_value=0, max_value=10, step=1)
    balance = st.number_input("Enter the customer's account balance:")
    num_of_products = st.number_input("Enter the number of bank products used by the customer (1-4):", min_value=1, max_value=4, step=1)
    has_cr_card = st.radio("Does the customer have a credit card?", (1, 0))
    is_active_member = st.radio("Is the customer an active member?", (1, 0))
    estimated_salary = st.number_input("Enter the customer's estimated salary:")

    # When the user clicks the "Predict" button
    if st.button('Predict'):
        # Create a DataFrame from the user input
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # Convert input data to the same format as training data
        input_data = pd.get_dummies(input_data, columns=['Geography', 'Gender'], drop_first=True)
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        input_features = sc.transform(input_data)

        # Predict churn
        churn_prediction = model.predict(input_features)
        churn_prediction_human_readable = np.where(churn_prediction == 1, 'Churn', 'No Churn')

        # Display the prediction
        st.write(f"The predicted churn status for the customer is: {churn_prediction_human_readable[0]}")

# Call the function to test
predict_churn_from_input()
