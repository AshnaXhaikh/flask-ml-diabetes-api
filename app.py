from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow requests from Vercel frontend

# Load saved model and preprocessor
model = joblib.load('log_reg.pkl')
preprocessor = joblib.load('data_preprocessor.pkl')

# Feature names in exact order your model expects
feature_names = ['age', 'hypertension', 'heart_disease', 'bmi',
                 'HbA1c_level', 'blood_glucose_level', 'gender_Male', 'is_smoker']

@app.route('/')
def home():
    return jsonify({"message": "Diabetes Prediction API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        input_values = []

        # Collect and validate input
        for f in feature_names:
            val = data.get(f)
            if val is None or str(val).strip() == "":
                return jsonify({"error": f"{f.replace('_', ' ')} is required"}), 400
            
            if f in ['hypertension', 'heart_disease', 'gender_Male', 'is_smoker']:
                input_values.append(int(val))
            else:
                input_values.append(float(val))

        input_df = pd.DataFrame([input_values], columns=feature_names)
        scaled_input = preprocessor.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({"prediction": result, "probability": round(prob*100, 2)})

    except ValueError as ve:
        return jsonify({"error": f"Invalid input. Error: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
