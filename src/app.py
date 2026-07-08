import streamlit as st
import pandas as pd
import joblib
import json
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(page_title="SmartHost", page_icon="🏨")
st.title("🏨 SmartHost")

@st.cache_resource
def load_assets():
    model = joblib.load('models/modelo_cancelacion.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/features.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, expected_features = load_assets()

with st.form("input_form"):
    st.subheader("Datos de la reserva")
    lead_time = st.number_input("Lead Time (días de antelación)", 0, 737, 30)
    adults = st.number_input("Adultos", 1, 10, 2)
    adr = st.number_input("Precio Promedio (ADR)", 0.0, 5000.0, 100.0)
    prev_cancellations = st.number_input("Cancelaciones previas del cliente", 0, 50, 0)
    special_requests = st.number_input("Peticiones especiales", 0, 10, 0)
    deposit = st.selectbox("Tipo de depósito", ["No Deposit", "Non Refund", "Refundable"])
    hotel = st.selectbox("Tipo de Hotel", ["City Hotel", "Resort Hotel"])
    submitted = st.form_submit_button("¡Predecir Reserva! 🚀")

if submitted:
    input_dict = {feature: 0 for feature in expected_features}
    input_dict['lead_time'] = lead_time
    input_dict['adults'] = adults
    input_dict['adr'] = adr
    input_dict['previous_cancellations'] = prev_cancellations
    input_dict['total_of_special_requests'] = special_requests
    
    if hotel == "City Hotel": input_dict['hotel_City Hotel'] = 1
    else: input_dict['hotel_Resort Hotel'] = 1
    
    dep_key = f"deposit_type_{deposit}"
    if dep_key in input_dict: input_dict[dep_key] = 1

    df_input = pd.DataFrame([input_dict])[expected_features]
    
    try:
        df_scaled = scaler.transform(df_input)
        prob = model.predict_proba(df_scaled)[0][1]
        
        st.write(f"### Probabilidad de cancelación: {prob:.2%}")
        
        if prob > 0.4:
            st.error("⚠️ Alerta: Riesgo alto. Esta reserva es probable que se cancele.")
        else:
            st.success("✅ Reserva segura.")
    except Exception as e:
        st.error(f"Error: {e}")

# Botón de reinicio fuera del form para mejor flujo
if st.button("Limpiar y reiniciar"):
    st.rerun()