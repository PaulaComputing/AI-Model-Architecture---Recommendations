import os
import time
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://api:8000")
API_KEY = os.getenv("API_KEY", "duoc-ep3-secret-key-2026")
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

st.set_page_config(page_title="E-commerce AI Recommender", layout="wide")
st.title("Sistema de Recomendación E-commerce (PyTorch + AWS EC2)")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Parámetros de Consulta")
    user_id = st.number_input("ID de Usuario", min_value=1, max_value=100000, value=42)
    top_k = st.slider("Cantidad de Recomendaciones (Top-K)", min_value=1, max_value=10, value=5)

    if st.button("Obtener Recomendaciones", type="primary"):
        try:
            res = requests.post(
                f"{API_URL}/api/v1/recommendations",
                json={"user_id": int(user_id), "top_k": int(top_k)},
                headers=HEADERS,
                timeout=5
            )
            if res.status_code == 200:
                st.session_state["last_rec"] = res.json()
            else:
                st.error(f"Error API ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"No se pudo conectar con la API: {e}")

    st.divider()
    st.subheader("Simular Evento de Usuario (RabbitMQ)")
    sku = st.text_input("SKU del Producto", value="SKU-1004")
    event_type = st.selectbox("Tipo de Interacción", ["view", "add_to_cart", "purchase"])
    if st.button("Enviar Evento a Cola"):
        try:
            ev = requests.post(
                f"{API_URL}/api/v1/events",
                json={"user_id": int(user_id), "product_id": sku, "event_type": event_type},
                headers=HEADERS,
                timeout=5
            )
            if ev.status_code == 200:
                st.success("Evento enviado a RabbitMQ y caché invalidada.")
            else:
                st.error(f"Error ({ev.status_code}): {ev.text}")
        except Exception as e:
            st.error(f"Error de conexión: {e}")

with col2:
    st.subheader("Resultados e Interpretabilidad del Modelo")
    if "last_rec" in st.session_state:
        data = st.session_state["last_rec"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Origen de Respuesta", data["source"])
        m2.metric("Latencia Real", f"{data['latency_ms']} ms", delta="OK (<200ms)")
        m3.metric("Usuario Consultado", f"#{data['user_id']}")

        df = pd.DataFrame(data["recommendations"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Presiona 'Obtener Recomendaciones' para consultar el modelo PyTorch.")