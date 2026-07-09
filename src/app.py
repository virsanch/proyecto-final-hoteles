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
    layout="wide"  # usamos todo el ancho para las dos columnas
)

# ============================================================
# ESTILOS PERSONALIZADOS (verde inglés + blanco)
# ============================================================
st.markdown("""
<style>
    /* Paleta verde inglés */
    :root {
        --verde-ingles: #1B4332;
        --verde-medio: #2D6A4F;
        --verde-claro: #40916C;
        --crema: #F8F9FA;
    }

    /* Fondo general */
    .stApp {
        background-color: #F8F9FA;
    }

    /* Banner del logo */
    .banner {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        text-align: center;
    }
    .banner h1 {
        color: white;
        font-size: 42px;
        margin: 0;
        font-weight: 700;
    }
    .banner p {
        color: #B7E4C7;
        font-size: 16px;
        margin: 8px 0 0 0;
    }

    /* Títulos de sección */
    .seccion-titulo {
        color: #1B4332;
        font-size: 24px;
        font-weight: 700;
        border-bottom: 3px solid #40916C;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* Botón principal */
    .stButton>button {
        background-color: #1B4332;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2D6A4F;
        color: white;
    }

    /* Tarjeta de recomendación */
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
# BANNER / LOGO
# ============================================================
st.markdown("""
<div class="banner">
    <h1>🏨 SmartHost</h1>
    <p>Sistema inteligente de predicción de cancelaciones hoteleras</p>
</div>
""", unsafe_allow_html=True)

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
        # Creamos el diccionario con las 245 variables a cero
        input_dict = {feature: 0 for feature in expected_features}

        # Rellenamos las variables que introdujo el usuario
        input_dict['lead_time'] = lead_time
        input_dict['adults'] = adults
        input_dict['adr'] = adr
        input_dict['previous_cancellations'] = prev_cancellations
        input_dict['total_of_special_requests'] = special_requests

        # Variables categóricas (OneHotEncoder)
        if hotel == "City Hotel":
            input_dict['hotel_City Hotel'] = 1
        else:
            input_dict['hotel_Resort Hotel'] = 1

        dep_key = f"deposit_type_{deposit}"
        if dep_key in input_dict:
            input_dict[dep_key] = 1

        # Creamos el DataFrame en el orden correcto
        df_input = pd.DataFrame([input_dict])[expected_features]

        try:
            # Escalamos y predecimos
            df_scaled = scaler.transform(df_input)
            prob = model.predict_proba(df_scaled)[0][1]
            porcentaje = prob * 100

            # ----------------------------------------------------
            # VELOCÍMETRO (gauge) con plotly
            # ----------------------------------------------------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=porcentaje,
                number={'suffix': "%", 'font': {'size': 40, 'color': '#1B4332'}},
                title={'text': "Probabilidad de cancelación", 'font': {'size': 18, 'color': '#1B4332'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#1B4332'},
                    'bar': {'color': '#1B4332'},
                    'steps': [
                        {'range': [0, 40], 'color': '#B7E4C7'},   # verde claro = bajo
                        {'range': [40, 70], 'color': '#FFE082'},  # amarillo = medio
                        {'range': [70, 100], 'color': '#EF9A9A'}  # rojo claro = alto
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

            # ----------------------------------------------------
            # RECOMENDACIÓN DE NEGOCIO según el riesgo
            # ----------------------------------------------------
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
        # Mensaje inicial cuando aún no se ha predicho
        st.info("Introduce los datos de la reserva y pulsa **Predecir cancelación** "
                "para ver el resultado.")

# ============================================================
# PIE DE PÁGINA
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6c757d; font-size:13px;'>"
    "SmartHost — Proyecto de Machine Learning · Modelo entrenado con 119.000 reservas reales"
    "</p>",
    unsafe_allow_html=True
)
