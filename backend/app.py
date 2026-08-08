import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Dynamically locate the model file within the backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "superkart_model.joblib")

# Load your model into memory once when the container boots up
try:
    model = joblib.load(MODEL_PATH)
    print("📢 Success: superkart_model.joblib successfully loaded!")
except Exception as e:
    print(f"❌ Critical Error loading model file: {e}")
    model = None


def engineer_features(df):
    """
    Applies custom feature engineering transformations.
    Automatically handles variations in column names (like Store_Age_Years)
    and falls back to smart matching if Product_Id is missing.
    """
    # Create a copy so we don't modify the original incoming data structure
    df_copy = df.copy()

    # -------------------------------------------------------------
    # 1. AUTOMATIC COLUMN RENAMING (CLEANUP STEP)
    # -------------------------------------------------------------
    # Standardize custom/incorrect column names sent by raw clients
    column_mapping = {
        "Store_Age_Years": "Store_Establishment_Year",
        "Product_Type_Category": "Product_Type"
    }
    # rename columns if they exist in the dataframe
    df_copy = df_copy.rename(columns=column_mapping)

    # -------------------------------------------------------------
    # 2. HANDLE STORE ESTABLISHMENT YEAR REVERSE MATH
    # -------------------------------------------------------------
    if "Store_Establishment_Year" in df_copy.columns:
        # Enforce integer data type conversion safely
        df_copy["Store_Establishment_Year"] = df_copy["Store_Establishment_Year"].astype(int)

        # If the client sent raw age (e.g. 17) instead of a year (e.g. 2009),
        # convert it back to a calendar year so our main logic works.
        if df_copy["Store_Establishment_Year"].max() < 150:
            df_copy["Store_Establishment_Year"] = 2026 - df_copy["Store_Establishment_Year"]

    # -------------------------------------------------------------
    # 3. CALCULATE STORE_AGE FEATURE
    # -------------------------------------------------------------
    if "Store_Establishment_Year" in df_copy.columns:
        df_copy["Store_Age"] = 2026 - df_copy["Store_Establishment_Year"]

    # -------------------------------------------------------------
    # 4. EXTRACT PRODUCT_BROAD_CATEGORY (WITH SMART FALLBACK)
    # -------------------------------------------------------------
    if "Product_Id" in df_copy.columns:
        # Strategy A: Use traditional ID extraction prefixes
        def map_category_by_id(pid):
            prefix = str(pid)[:2].upper()
            if prefix == "FD": return "Food"
            elif prefix == "DR": return "Drinks"
            else: return "Non-Consumable"
        df_copy["Product_Broad_Category"] = df_copy["Product_Id"].apply(map_category_by_id)

    elif "Product_Type" in df_copy.columns:
        # Strategy B: Fallback text mapping because Product_Id is missing entirely
        def map_category_by_text(p_type):
            val = str(p_type).lower()
            if any(word in val for word in ["food", "snack", "baking", "meat", "dairy", "fruits", "bread"]):
                return "Food"
            elif any(word in val for word in ["drink", "beverage", "soft"]):
                return "Drinks"
            else:
                return "Non-Consumable"
        df_copy["Product_Broad_Category"] = df_copy["Product_Type"].apply(map_category_by_text)

    else:
        # Strategy C: Hard fallback if everything is missing to avoid model crash
        df_copy["Product_Broad_Category"] = "Food"
        df_copy["Product_Type"] = "Snack Foods"
    return df_copy

# =============================================================
# VERSION 1: SINGLE PREDICTION ENDPOINT
# =============================================================
@app.route("/v1/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file missing or not loaded on server."}), 500

    try:
        json_data = request.get_json()
        raw_df = pd.DataFrame([json_data])
        engineered_df = engineer_features(raw_df)
        final_input = engineered_df.drop(
            columns=['Product_Store_Sales_Total', 'Product_Id', 'Store_Id'],
            errors='ignore'
        )
        prediction_array = model.predict(final_input)
        final_prediction = float(prediction_array[0])

        return jsonify({"prediction": final_prediction}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to compute prediction: {str(e)}"}), 400


# =============================================================
# VERSION 1: BATCH PREDICTION ENDPOINT
# =============================================================
@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model file missing or not loaded on server."}), 500

    try:
        json_data = request.get_json()

        if not isinstance(json_data, list):
            return jsonify({"error": "Batch data must be sent as a JSON list array."}), 400

        raw_df = pd.DataFrame(json_data)
        engineered_df = engineer_features(raw_df)
        final_input = engineered_df.drop(
            columns=['Product_Store_Sales_Total', 'Product_Id', 'Store_Id'],
            errors='ignore'
        )
        prediction_array = model.predict(final_input)
        final_predictions = [float(val) for val in prediction_array]

        return jsonify({"predictions": final_predictions}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to compute batch predictions: {str(e)}"}), 400


@app.route("/", methods=["GET"])
def home():
    status = "loaded successfully" if model is not None else "failed to load"
    return jsonify({
        "status": "Backend API server is up and running!",
        "version": "v1",
        "model_status": status
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
