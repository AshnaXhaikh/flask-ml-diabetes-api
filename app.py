from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and preprocessor
model = joblib.load('log_reg.pkl')
preprocessor = joblib.load('data_preprocessor.pkl')

# Feature names in exact order your model expects
# This list must not change, as it matches the model's training
feature_names = ['age', 'hypertension', 'heart_disease', 'bmi',
                 'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'is_smoker']

@app.route('/')
def home():
    # Pass the feature names to the template to dynamically build the form
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = []

        # Collect and validate user inputs
        for f in feature_names:
            val = request.form.get(f)
            if val is None or val.strip() == "":
                # Handle missing value error
                return render_template('result.html', prediction=f"Error: {f.replace('_', ' ')} is required!", probability=0)
            
            # REFINEMENT:
            # Since the new HTML form sends "1" or "0" for binary features,
            # we can just safely cast them to int.
            if f in ['hypertension', 'heart_disease', 'gender_Male', 'is_smoker']:
                input_values.append(int(val))
            else:
                # Continuous features
                input_values.append(float(val))

        # Create DataFrame with correct column names
        input_df = pd.DataFrame([input_values], columns=feature_names)

        # Apply preprocessor (scaling)
        scaled_input = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1] # Probability of diabetic

        result = "🩸 Diabetic" if prediction == 1 else "💚 Non-Diabetic"
        return render_template('result.html', prediction=result, probability=round(prob*100, 2))

    except ValueError as ve:
        # Handle cases where conversion to float/int fails
        return render_template('result.html', prediction=f"Invalid input. Please check your values. Error: {ve}", probability=0)
    except Exception as e:
        # General error handler
        return render_template('result.html', prediction=f"An error occurred: {e}", probability=0)


if __name__ == '__main__':
    app.run(debug=True)