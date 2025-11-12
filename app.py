import os
import pickle
import pandas as pd
# IMPORTANT: Use Flask imports, including render_template and jsonify
from flask import Flask, render_template, request, jsonify 

# --- Configuration ---
# Ensure correct relative paths from the app/main.py location
# The app is run from the project root, so the path is relative to the root, not app/.
MODEL_PATH = os.path.join("models", "diabetes_model.pkl")
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")

# Initialize Flask app
# The template_folder is set to look in 'app/templates' relative to the app file.
# Railway's build process usually runs the app from the root directory.
app = Flask(__name__, template_folder='app/templates') 

# Load Model and Preprocessor
model = None
preprocessor = None

# --- Model Loading ---
def load_models():
    """Loads the ML model and preprocessor."""
    global model, preprocessor

    # Check if model files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        print(f"ERROR: Model files not found. Ensure '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' exist.")
        return

    try:
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        print("--- ML Model and Preprocessor loaded successfully! ---")

    except Exception as e:
        print(f"ERROR: Failed to load models: {e}")
        model = None
        preprocessor = None

# Load models immediately when the app starts
load_models()

# --- Endpoints ---

@app.route("/", methods=["GET"])
def home():
    """Serve the main prediction HTML page using Flask's render_template."""
    # This should now correctly resolve to app/templates/index.html
    return render_template("index.html")


# You can remove this simple health check if you don't need it, 
# as the / route now serves HTML.
# @app.route("/api/health")
# def read_root():
#     return jsonify({"message": "Diabetes Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission, preprocess data, and return prediction page."""
    if model is None or preprocessor is None:
        return render_template('result.html', prediction="Model not loaded!", probability=0)

    try:
        # Collect form data instead of JSON
        form_data = request.form

        # Prepare input values
        COLUMNS = ['age', 'hypertension', 'heart_disease', 'bmi', 
                   'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'is_smoker']
        
        input_values = []
        for key in COLUMNS:
            val = form_data.get(key)
            if val is None or val.strip() == "":
                return render_template('result.html', prediction=f"{key} is required", probability=0)
            
            if key in ['hypertension', 'heart_disease', 'gender_Male', 'is_smoker']:
                input_values.append(int(val))
            else:
                input_values.append(float(val))
        
        # Create DataFrame and preprocess
        raw_data = pd.DataFrame([input_values], columns=COLUMNS)
        processed_data = preprocessor.transform(raw_data)

        # Make prediction
        prediction_class = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]

        # Format result
        result_text = "🩸 Diabetic" if prediction_class == 1 else "💚 Non-Diabetic"
        probability = round(prediction_proba[1] * 100, 2)

        return render_template('result.html', prediction=result_text, probability=probability)

    except Exception as e:
        return render_template('result.html', prediction=f"An error occurred: {e}", probability=0)


# Railway will use Gunicorn/Waitress, which calls the 'app' instance directly.
# The code below is only for local testing.
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8000)

