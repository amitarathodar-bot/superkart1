import os
import io
import pandas as pd
import streamlit as st
import requests

st.set_page_config(page_title="Superkart Sales Predictor", layout="wide")
st.title("🛒 Superkart Store Sales Prediction Dashboard")

# Look for Docker's internal network address first; fallback to localhost if testing outside Docker
BACKEND_URL = "http://backend:7860/v1/predict"
# Create structural tabs for the two inference modes
tab1, tab2 = st.tabs(["🎯 Single Prediction (Online)", "📦 Bulk Upload (Batch)"])

# ==========================================
# TAB 1: ONLINE INFERENCE (SINGLE PREDICTION)
# ==========================================
with tab1:
    st.subheader("Predict Sales for a Single Product")

    col1, col2, col3 = st.columns(3)

    with col1:
        prod_id = st.text_input("Product ID", "FD6114")
        weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        area = st.number_input("Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.027, format="%.4f")

    with col2:
        p_type = st.selectbox("Product Type", ["Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene", "Snack Foods"])
        mrp = st.number_input("Product MRP ($)", min_value=0.0, value=117.08)
        store_id = st.text_input("Store ID", "OUT004")
        year = st.number_input("Establishment Year", min_value=1900, max_value=2026, value=2009)

    with col3:
        size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        city = st.selectbox("City Location Tier", ["Tier 1", "Tier 2", "Tier 3"])
        s_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

    if st.button("🔮 Forecast Single Sales", type="primary"):
        # Match the raw 11 features structure expected by the backend
        payload = {
            "Product_Id": prod_id,
            "Product_Weight": weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area,
            "Product_Type": p_type,
            "Product_MRP": mrp,
            "Store_Id": store_id,
            "Store_Establishment_Year": int(year),
            "Store_Size": size,
            "Store_Location_City_Type": city,
            "Store_Type": s_type
        }

        try:
            with st.spinner("Calculating prediction..."):
                response = requests.post(BACKEND_URL, json=payload)
                res_data = response.json()

            if response.status_code == 200:
                st.success(f"🎉 Predicted Product Store Sales: **${res_data['prediction']:.2f}**")
            else:
                st.error(f"Backend Error: {res_data.get('error')}")
        except Exception as e:
            st.error(f"Could not connect to Backend API: {e}")

# ==========================================
# TAB 2: BATCH INFERENCE
# ==========================================
with tab2:
    st.subheader("Predict Sales for Multiple Products via CSV")
    st.write("Upload a CSV file containing the 11 raw features columns to process batch predictions.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("📋 **Preview of Uploaded Data:**", input_df.head())

            if st.button("🚀 Process Batch Predictions"):
                predictions = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Iterate rows and send individually to the REST API endpoint
                total_rows = len(input_df)
                for index, row in input_df.iterrows():
                    payload = row.to_dict()

                    # Convert data types safely to avoid JSON serialization errors
                    if 'Store_Establishment_Year' in payload:
                        payload['Store_Establishment_Year'] = int(payload['Store_Establishment_Year'])

                    try:
                        response = requests.post(BACKEND_URL, json=payload)
                        if response.status_code == 200:
                            predictions.append(response.json()["prediction"])
                        else:
                            predictions.append(None)
                    except:
                        predictions.append(None)

                    # Update progress UI elements
                    progress = (index + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Processing row {index + 1} of {total_rows}...")

                # Append predictions back to dataframe
                input_df["Predicted_Store_Sales"] = predictions
                status_text.empty()
                st.success("✅ Batch Inference Complete!")
                st.write("📊 **Results Preview:**", input_df.head())

                # Provide download utility button
                csv_buffer = io.StringIO()
                input_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_buffer.getvalue(),
                    file_name="superkart_sales_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error parsing file: {e}")
