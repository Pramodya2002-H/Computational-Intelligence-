from flask import Flask, render_template, request
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug: Print received form data
        print("Received form data:", request.form)
        # Extract and validate input from HTML form
        if not request.form:
            return "Error: No data submitted", 400

        applicant_income = request.form.get('applicant_income')
        person_home_ownership = request.form.get('person_home_ownership')
        loan_amnt = request.form.get('loan_amnt')
        loan_int_rate = request.form.get('loan_int_rate')
        cb_person_cred_hist_length = request.form.get('cb_person_cred_hist_length')
        person_age = request.form.get('person_age')
        loan_grade = request.form.get('loan_grade')
        cb_person_default_on_file = request.form.get('cb_person_default_on_file')
        loan_intent = request.form.get('loan_intent')
        person_emp_length = request.form.get('person_emp_length')
        loan_percent_income = request.form.get('loan_percent_income')

        # Validate and convert numeric fields
        try:
            applicant_income = float(applicant_income) if applicant_income else 0.0
            loan_amnt = float(loan_amnt) if loan_amnt else 0.0
            loan_int_rate = float(loan_int_rate) if loan_int_rate else 0.0
            cb_person_cred_hist_length = int(cb_person_cred_hist_length) if cb_person_cred_hist_length else 0
            person_age = int(person_age) if person_age else 0
            person_emp_length = int(person_emp_length) if person_emp_length else 0
            loan_percent_income = float(loan_percent_income) if loan_percent_income else 0.0
        except ValueError:
            return "Error: Please enter valid numeric values for income, amount, rate, length, age, and percentage.", 400

        # Check for required categorical fields
        categorical_fields = [person_home_ownership, loan_grade, cb_person_default_on_file, loan_intent]
        if not all(categorical_fields):
            missing = [field for field in ['person_home_ownership', 'loan_grade', 'cb_person_default_on_file', 'loan_intent'] if not request.form.get(field)]
            return f"Error: Please fill out all categorical fields. Missing: {', '.join(missing)}", 400

        # Convert to DataFrame
        input_df = pd.DataFrame({
            'person_home_ownership': [person_home_ownership],
            'person_income': [applicant_income],
            'loan_int_rate': [loan_int_rate],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length],
            'loan_amnt': [loan_amnt],
            'person_age': [person_age],
            'loan_grade': [loan_grade],
            'cb_person_default_on_file': [cb_person_default_on_file],
            'loan_intent': [loan_intent],
            'person_emp_length': [person_emp_length],
            'loan_percent_income': [loan_percent_income]
        })

        # Predict
        prediction = model.predict(input_df)[0]

        result = "Approved " if prediction == 1 else "Rejected "

        return render_template('result.html', result=result)
    except Exception as e:
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
    