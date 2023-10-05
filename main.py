import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('HR_Dataset.csv')

# Separate features (X) and target (y)
X = dataset.drop(columns=['left']).values
y = dataset['left'].values

# Perform one-hot encoding for categorical columns (columns 7 and 8)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7, 8])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize numeric features (exclude one-hot encoded features)
numeric_features = [0, 1, 2, 3, 4, 5]
sc = StandardScaler()
X_train[:, numeric_features] = sc.fit_transform(X_train[:, numeric_features])
X_test[:, numeric_features] = sc.transform(X_test[:, numeric_features])

# Train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)


# Function to display the confusion matrix and accuracy score
def display_results(y_test, y_pred):
    st.subheader("Model Evaluation")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    st.text(f"Accuracy: {accuracy:.2f}")


# Streamlit app
st.title("Employee Churn Prediction using Random Forest Classification")
st.text(" ")
st.text(" ")
st.text(" ")
st.image('employees.jpg', width=700)
st.text(" ")
st.text(" ")
st.text(" ")

# Display images or explanatory text as needed
# st.image('employees.jpeg', width=700)
st.text("")

# Display bar charts for categorical features
st.subheader("Categorical Feature Analysis")
st.text("")

# Bar chart for Department vs. Churn
st.text("Department vs. Churn")
department_churn = dataset.groupby('Departments ')['left'].value_counts().unstack().fillna(0)
department_churn.plot(kind='bar', stacked=True)
plt.xlabel("Department")
plt.ylabel("Count")
plt.title("Employee Churn by Department")
st.pyplot()

# Bar chart for Salary vs. Churn
st.text("Salary vs. Churn")
department_churn = dataset.groupby('salary')['left'].value_counts().unstack().fillna(0)
department_churn.plot(kind='bar', stacked=True)
plt.xlabel("Salary")
plt.ylabel("Count")
plt.title("Employee Churn by Department")
st.pyplot()

st.text("")

# Display pair plots for numerical features
st.subheader("Pair Plots for Numerical Features")
sns.pairplot(dataset, hue="left")
st.pyplot()

st.text("")

st.title("Predict Employee Churn")

# User inputs for categorical features
department = st.selectbox("Select Department", dataset['Departments '].unique())
salary = st.selectbox("Select Salary", dataset['salary'].unique())

# Map the selected values to the one-hot encoding used during training
departments_unique = dataset['Departments '].unique()
salaries_unique = dataset['salary'].unique()
department_encoded = np.zeros(len(departments_unique))
salary_encoded = np.zeros(len(salaries_unique))

for i, d in enumerate(departments_unique):
    if d == department:
        department_encoded[i] = 1

for i, s in enumerate(salaries_unique):
    if s == salary:
        salary_encoded[i] = 1

# User inputs for other numeric features
satisfaction_level = st.number_input("Enter Satisfaction Level (between 0 and 1)")
last_evaluation = st.number_input("Enter Last Evaluation (between 0 and 1)")
number_project = st.number_input("Enter Number of Projects")
average_montly_hours = st.number_input("Enter Average Monthly Hours")
time_spend_company = st.number_input("Enter the Time spent in Company")
Work_accident = st.number_input("Enter Work Accident")
promotion_last_5years = st.number_input("Enter Promotion (0 or 1)")

if st.button("Predict"):
    # Concatenate the one-hot encoded features with other features
    user_inputs = np.concatenate([department_encoded, salary_encoded,
                                  [satisfaction_level, last_evaluation, number_project, average_montly_hours,
                                   time_spend_company, Work_accident, promotion_last_5years]])

    # Make a prediction
    prediction = classifier.predict([user_inputs])[0]

    st.subheader("Predicted Employee Churn")
    if prediction == 0:
        st.write("Not Churned")
    else:
        st.write("Churned")

st.text("")
# Display data visualization
st.subheader("Data Visualization")
st.text("")
st.text("Relationship between Satisfaction Level and Employee Churn")
plt.figure(figsize=(9, 6))
plt.xlabel("Satisfaction Level")
plt.ylabel("Employee Churn")
plt.scatter(dataset['satisfaction_level'], dataset['left'], color='r')
plt.tight_layout()
st.pyplot()

st.text("")
st.text("Relationship between Last Evaluation and Employee Churn")
plt.figure(figsize=(9, 6))
plt.xlabel("Last Evaluation")
plt.ylabel("Employee Churn")
plt.scatter(dataset['last_evaluation'], dataset['left'], color='r')
plt.tight_layout()
st.pyplot()

st.text("")
st.text("Relationship between Number of Projects and Employee Churn")
plt.figure(figsize=(9, 6))
plt.xlabel("Number of Projects")
plt.ylabel("Employee Churn")
plt.scatter(dataset['number_project'], dataset['left'], color='r')
plt.tight_layout()
st.pyplot()

st.text("")
st.text("Relationship between Average Monthly Hours and Employee Churn")
plt.figure(figsize=(9, 6))
plt.xlabel("Average Monthly Hours")
plt.ylabel("Employee Churn")
plt.scatter(dataset['average_montly_hours'], dataset['left'], color='r')
plt.tight_layout()
st.pyplot()

st.text("")
st.text("Relationship between Time spent in Company and Employee Churn")
plt.figure(figsize=(9, 6))
plt.xlabel("Time spent in Company")
plt.ylabel("Employee Churn")
plt.scatter(dataset['time_spend_company'], dataset['left'], color='r')
plt.tight_layout()
st.pyplot()

st.text("")
st.text("Relationship between Work Accident and Employee Churn")
plt.figure(figsize=(9, 6))
plt.xlabel("Work Accident")
plt.ylabel("Employee Churn")
plt.scatter(dataset['Work_accident'], dataset['left'], color='r')
plt.tight_layout()
st.pyplot()

st.text("")
st.text("Relationship between Promotion and Employee Churn")
plt.figure(figsize=(9, 6))
plt.xlabel("Promotion")
plt.ylabel("Employee Churn")
plt.scatter(dataset['promotion_last_5years'], dataset['left'], color='r')
plt.tight_layout()
st.pyplot()

st.set_option('deprecation.showPyplotGlobalUse', False)
