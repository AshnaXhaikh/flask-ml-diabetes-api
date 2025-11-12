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
    """Handle JSON submission, preprocess data, and return JSON prediction."""
    if model is None or preprocessor is None:
        return jsonify(
            {"error": "Prediction service is unavailable. Model files could not be loaded."}
        ), 500

    try:
        # **FLASK JSON CHANGE: Get data from the JSON body**
        form_data = request.get_json()
        
        if not form_data:
            return jsonify({"error": "No JSON data received. Check Content-Type header."}), 400
            
        # 1. Collect data into a DataFrame. Ensure types are correct.
        COLUMNS = [
            'age', 'hypertension', 'heart_disease', 'bmi', 
            'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'is_smoker'
        ]
        
        # Prepare input values, converting them to the correct type based on keys
        input_values = []
        for key in COLUMNS:
            val = form_data.get(key)
            if val is None or val == "":
                raise ValueError(f"Missing required field: {key}")

            if key in ['hypertension', 'heart_disease', 'gender_Male', 'is_smoker']:
                # Binary features are integers
                input_values.append(int(val))
            else:
                # Continuous features are floats
                input_values.append(float(val))

        raw_data = pd.DataFrame([input_values], columns=COLUMNS)

        # 2. Preprocess the data
        processed_data = preprocessor.transform(raw_data)

        # 3. Make Prediction
        prediction_class = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]

        # 4. Format results
        result = {
            "prediction": int(prediction_class), # 0 for No Diabetes, 1 for Diabetes
            "probability_no_diabetes": round(prediction_proba[0] * 100, 2),
            "probability_diabetes": round(prediction_proba[1] * 100, 2),
            "message": "Prediction successful."
        }

        # Flask returns JSON using jsonify
        return jsonify(result)

    except ValueError as ve:
        # Handle cases where inputs could not be converted to int/float
        print(f"Input conversion error: {ve}")
        return jsonify(
            {"error": f"Invalid input value: {ve}"}
        ), 400

    except Exception as e:
        print(f"An unexpected server error occurred: {e}")
        # The generic error handling returns JSON
        return jsonify(
            {"error": f"An unexpected server error occurred."}
        ), 500

# Railway will use Gunicorn/Waitress, which calls the 'app' instance directly.
# The code below is only for local testing.
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8000)
