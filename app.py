import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

# --- Configuration ---
MODEL_PATH = "log_reg.pkl"            # Model in root directory
PREPROCESSOR_PATH = "data_preprocessor.pkl"  # Preprocessor in root directory

# Initialize Flask app
app = Flask(__name__)

# Load Model and Preprocessor
model = None
preprocessor = None

def load_models():
    """Loads the ML model and preprocessor."""
    global model, preprocessor

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print(f"ERROR: Model files not found. Check '{MODEL_PATH}' and '{PREPROCESSOR_PATH}'")
        return

    try:
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("--- ML Model and Preprocessor loaded successfully! ---")
    except Exception as e:
        print(f"ERROR loading models: {e}")
        model = None
        preprocessor = None

# Load models at startup
load_models()

# --- Routes ---

@app.route("/", methods=["GET"])
def home():
    # Serve the main form page
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or preprocessor is None:
        return render_template('result.html', prediction="Model not loaded!", probability=0)

    try:
        feature_names = [
            'age', 'hypertension', 'heart_disease', 'bmi',
            'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'is_smoker'
        ]
        input_values = []

        # Collect form data
        for f in feature_names:
            val = request.form.get(f)
            if val is None or val.strip() == "":
                return render_template(
                    'result.html', 
                    prediction=f"Error: {f.replace('_',' ')} is required!", 
                    probability=0
                )

            # Convert types
            if f in ['hypertension', 'heart_disease', 'gender_Male', 'is_smoker']:
                input_values.append(int(val))
            else:
                input_values.append(float(val))

        # Convert to DataFrame
        input_df = pd.DataFrame([input_values], columns=feature_names)
        processed_input = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]

        # Format result for result.html
        result_text = "🩸 Diabetic" if prediction == 1 else "💚 Non-Diabetic"
        probability_percent = round(probability * 100, 2)

        return render_template('result.html', prediction=result_text, probability=probability_percent)

    except Exception as e:
        return render_template('result.html', prediction=f"An error occurred: {e}", probability=0)
