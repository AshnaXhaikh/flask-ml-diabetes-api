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
app = Flask(__name__) 

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
    return render_template("index.html", features=[
        'age', 'hypertension', 'heart_disease', 'bmi',
        'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'is_smoker'
    ])



@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_values = []
        feature_names = ['age', 'hypertension', 'heart_disease', 'bmi',
                         'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'is_smoker']

        for f in feature_names:
            val = request.form.get(f)
            if val is None or val.strip() == "":
                return render_template('result.html', prediction=f"Error: {f.replace('_', ' ')} is required!", probability=0)

            if f in ['hypertension', 'heart_disease', 'gender_Male', 'is_smoker']:
                input_values.append(int(val))
            else:
                input_values.append(float(val))

        input_df = pd.DataFrame([input_values], columns=feature_names)
        scaled_input = preprocessor.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        # Show result in the format you want
        result_text = "🩸 Diabetic" if prediction == 1 else "💚 Non-Diabetic"
        probability = round(prob*100, 2)

        return render_template('result.html', prediction=result_text, probability=probability)

    except Exception as e:
        return render_template('result.html', prediction=f"An error occurred: {e}", probability=0)



# Railway will use Gunicorn/Waitress, which calls the 'app' instance directly.
# The code below is only for local testing.
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8000)



