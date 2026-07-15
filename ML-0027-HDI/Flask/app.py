"""
Human Development Index (HDI) Predictor - Flask Web Application
------------------------------------------------------------------
Loads a trained Linear Regression model (HDI.pkl) and serves a web
interface + REST API for predicting a country's HDI score based on
life expectancy, schooling, and income indicators.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ------------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "HDI.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def classify_hdi(score: float) -> str:
    """Classify a numeric HDI score into its development tier."""
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"


def validate_inputs(life_exp, mean_school, exp_school, gni):
    """Validate that submitted indicator values fall within realistic ranges."""
    errors = []
    if not (0 <= life_exp <= 100):
        errors.append("Life expectancy must be between 0 and 100 years.")
    if not (0 <= mean_school <= 20):
        errors.append("Mean years of schooling must be between 0 and 20.")
    if not (0 <= exp_school <= 25):
        errors.append("Expected years of schooling must be between 0 and 25.")
    if not (0 < gni <= 200000):
        errors.append("GNI per capita must be a positive value up to 200,000.")
    return errors


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.route("/")
def home():
    """Home page with project introduction."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """HDI prediction form page."""
    prediction = None
    category = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            life_exp = float(request.form.get("life_expectancy", ""))
            mean_school = float(request.form.get("mean_schooling", ""))
            exp_school = float(request.form.get("expected_schooling", ""))
            gni = float(request.form.get("gni_per_capita", ""))

            form_data = {
                "life_expectancy": life_exp,
                "mean_schooling": mean_school,
                "expected_schooling": exp_school,
                "gni_per_capita": gni,
            }

            errors = validate_inputs(life_exp, mean_school, exp_school, gni)
            if errors:
                error = " ".join(errors)
            else:
                features = pd.DataFrame(
                    [[life_exp, mean_school, exp_school, gni]],
                    columns=["Life_Expectancy", "Mean_Years_Schooling",
                             "Expected_Years_Schooling", "GNI_Per_Capita"],
                )
                raw_pred = float(model.predict(features)[0])
                raw_pred = max(0.0, min(1.0, raw_pred))  # clamp to valid HDI range
                prediction = round(raw_pred, 3)
                category = classify_hdi(prediction)

        except (ValueError, TypeError):
            error = "Please enter valid numeric values for all fields."

    return render_template(
        "predict.html",
        prediction=prediction,
        category=category,
        error=error,
        form_data=form_data,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON REST API endpoint for programmatic HDI prediction."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body."}), 400

    required_fields = [
        "life_expectancy",
        "mean_schooling",
        "expected_schooling",
        "gni_per_capita",
    ]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        life_exp = float(data["life_expectancy"])
        mean_school = float(data["mean_schooling"])
        exp_school = float(data["expected_schooling"])
        gni = float(data["gni_per_capita"])
    except (ValueError, TypeError):
        return jsonify({"error": "All fields must be numeric."}), 400

    errors = validate_inputs(life_exp, mean_school, exp_school, gni)
    if errors:
        return jsonify({"error": " ".join(errors)}), 400

    features = pd.DataFrame(
        [[life_exp, mean_school, exp_school, gni]],
        columns=["Life_Expectancy", "Mean_Years_Schooling",
                 "Expected_Years_Schooling", "GNI_Per_Capita"],
    )
    raw_pred = float(model.predict(features)[0])
    raw_pred = max(0.0, min(1.0, raw_pred))
    score = round(raw_pred, 3)
    category = classify_hdi(score)

    return jsonify({
        "hdi_score": score,
        "hdi_category": category,
        "inputs": {
            "life_expectancy": life_exp,
            "mean_schooling": mean_school,
            "expected_schooling": exp_school,
            "gni_per_capita": gni,
        },
    })


@app.route("/about")
def about():
    """About page describing the HDI methodology."""
    return render_template("about.html")


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
