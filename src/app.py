import streamlit as st
import pandas as pd
import joblib
import json
import warnings
import plotly.graph_objects as go
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="SmartHost",
    page_icon="🏨",
    layout="wide"
)

# ============================================================
# ESTILOS PERSONALIZADOS (verde inglés + oro viejo + crema)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #F8F9FA;
    }

    .banner {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
        padding: 35px 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .banner h1 {
        font-family: 'Playfair Display', serif;
        color: #F4EAD5;
        font-size: 46px;
        margin: 15px 0 0 0;
        font-weight: 500;
        letter-spacing: 2px;
    }
    .banner p {
        font-family: 'Playfair Display', serif;
        color: #B7E4C7;
        font-size: 15px;
        margin: 6px 0 0 0;
        font-style: italic;
        letter-spacing: 1px;
    }

    .seccion-titulo {
        color: #1B4332;
        font-size: 24px;
        font-weight: 700;
        border-bottom: 3px solid #C5A028;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    .stButton>button {
        background-color: #1B4332;
        color: #F4EAD5;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2D6A4F;
        color: #F4EAD5;
    }

    .recomendacion {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 16px;
    }
    .rec-alto {
        background-color: #FFEBEE;
        border-left: 5px solid #C62828;
        color: #C62828;
    }
    .rec-medio {
        background-color: #FFF8E1;
        border-left: 5px solid #F9A825;
        color: #F57F17;
    }
    .rec-bajo {
        background-color: #E8F5E9;
        border-left: 5px solid #2E7D32;
        color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# BANNER CON LOGO DE SELLO DE LACRE (SVG dorado)
# ============================================================
sello_svg = '<svg width="90" height="90" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="46" fill="none" stroke="#C5A028" stroke-width="1.5"/><circle cx="50" cy="50" r="40" fill="none" stroke="#C5A028" stroke-width="1"/><circle cx="50" cy="50" r="37" fill="none" stroke="#C5A028" stroke-width="0.6"/><g stroke="#C5A028" stroke-width="1"><line x1="50" y1="4" x2="50" y2="10"/><line x1="50" y1="90" x2="50" y2="96"/><line x1="4" y1="50" x2="10" y2="50"/><line x1="90" y1="50" x2="96" y2="50"/><line x1="18" y1="18" x2="22" y2="22"/><line x1="78" y1="78" x2="82" y2="82"/><line x1="18" y1="82" x2="22" y2="78"/><line x1="78" y1="22" x2="82" y2="18"/></g><text x="50" y="62" font-family="Georgia, serif" font-size="34" font-weight="600" fill="#C5A028" text-anchor="middle" letter-spacing="1">SH</text></svg>'

banner_html = '<div class="banner">' + sello_svg + '<h1>SmartHost</h1><p>Sistema inteligente de predicción de cancelaciones hoteleras</p></div>'

st.markdown(banner_html, unsafe_allow_html=True)

# ============================================================
# CARGAR MODELO, ESCALADOR Y FEATURES
# ============================================================
@st.cache_resource
def load_assets():
    model = joblib.load('models/modelo_cancelacion.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/features.json', 'r') as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, expected_features = load_assets()

# ============================================================
# DISEÑO EN DOS COLUMNAS
# ============================================================
col_izq, col_der = st.columns([1, 1])

# ------------------------------------------------------------
# COLUMNA IZQUIERDA — FORMULARIO
# ------------------------------------------------------------
with col_izq:
    st.markdown('<p class="seccion-titulo">Datos de la reserva</p>', unsafe_allow_html=True)

    with st.form("input_form"):
        lead_time = st.number_input("Días de antelación", 0, 737, 30,
                                    help="Días entre la reserva y la llegada")
        adults = st.number_input("Número de adultos", 1, 10, 2)
        adr = st.number_input("Precio por noche (€)", 0.0, 5000.0, 100.0)
        prev_cancellations = st.number_input("Cancelaciones previas del cliente", 0, 50, 0)
        special_requests = st.number_input("Peticiones especiales", 0, 10, 0)
        deposit = st.selectbox("Tipo de depósito",
                               ["No Deposit", "Non Refund", "Refundable"])
        hotel = st.selectbox("Tipo de hotel",
                             ["City Hotel", "Resort Hotel"])

        submitted = st.form_submit_button("Predecir cancelación")

# ------------------------------------------------------------
# COLUMNA DERECHA — RESULTADO
# ------------------------------------------------------------
with col_der:
    st.markdown('<p class="seccion-titulo">Resultado de la predicción</p>', unsafe_allow_html=True)

    if submitted:
        input_dict = {feature: 0 for feature in expected_features}

        input_dict['lead_time'] = lead_time
        input_dict['adults'] = adults
        input_dict['adr'] = adr
        input_dict['previous_cancellations'] = prev_cancellations
        input_dict['total_of_special_requests'] = special_requests

        if hotel == "City Hotel":
            input_dict['hotel_City Hotel'] = 1
        else:
            input_dict['hotel_Resort Hotel'] = 1

        dep_key = f"deposit_type_{deposit}"
        if dep_key in input_dict:
            input_dict[dep_key] = 1

        df_input = pd.DataFrame([input_dict])[expected_features]

        try:
            df_scaled = scaler.transform(df_input)
            prob = model.predict_proba(df_scaled)[0][1]
            porcentaje = prob * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=porcentaje,
                number={'suffix': "%", 'font': {'size': 40, 'color': '#1B4332'}},
                title={'text': "Probabilidad de cancelación", 'font': {'size': 18, 'color': '#1B4332'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#1B4332'},
                    'bar': {'color': '#1B4332'},
                    'steps': [
                        {'range': [0, 40], 'color': '#B7E4C7'},
                        {'range': [40, 70], 'color': '#FFE082'},
                        {'range': [70, 100], 'color': '#EF9A9A'}
                    ],
                    'threshold': {
                        'line': {'color': '#C62828', 'width': 4},
                        'thickness': 0.75,
                        'value': porcentaje
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20),
                              paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            if porcentaje >= 70:
                st.markdown("""
                <div class="recomendacion rec-alto">
                    <strong>⚠️ Riesgo alto de cancelación</strong><br>
                    Recomendamos solicitar un depósito o contactar al cliente
                    para confirmar la reserva. Considere aplicar overbooking controlado.
                </div>
                """, unsafe_allow_html=True)
            elif porcentaje >= 40:
                st.markdown("""
                <div class="recomendacion rec-medio">
                    <strong>🔶 Riesgo medio de cancelación</strong><br>
                    Sugerimos enviar un recordatorio de confirmación al cliente
                    unos días antes de la llegada.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recomendacion rec-bajo">
                    <strong>✅ Reserva fiable</strong><br>
                    Baja probabilidad de cancelación. No se requiere ninguna
                    acción especial para esta reserva.
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")

    else:
        st.info("Introduce los datos de la reserva y pulsa **Predecir cancelación** "
                "para ver el resultado.")

# ============================================================
# BOTÓN DE LIMPIAR Y REINICIAR (fuera del formulario para mejor flujo)
# ============================================================
st.markdown("---")
if st.button("Limpiar y reiniciar"):
    st.rerun()

# ============================================================
# PIE DE PÁGINA
# ============================================================
st.markdown(
    "<p style='text-align:center; color:#6c757d; font-size:13px;'>"
    "SmartHost — Proyecto de Machine Learning · Modelo entrenado con 119.000 reservas reales"
    "</p>",
    unsafe_allow_html=True
)
