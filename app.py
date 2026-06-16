from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from app_utils import CLASS_NAMES, CHECKPOINT_PATH, draw_predictions, load_model, run_inference

st.set_page_config(
    page_title="Detección de daños en autos",
    page_icon="🚗",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuración")
    score_threshold = st.slider(
        "Umbral de confianza",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        help="Solo se muestran detecciones con score mayor a este valor.",
    )

    st.divider()
    st.subheader("Modelo")
    st.markdown(
        """
        - **Arquitectura**: Faster R-CNN MobileNet V3 Large FPN
        - **Dataset**: CarDD (Car Damage Detection)
        - **Clases**: 6 tipos de daño
        - **mAP@50**: 0.648
        - **mAP@50:95**: 0.447
        """
    )

    st.divider()
    st.subheader("Clases detectadas")
    for class_id, name in CLASS_NAMES.items():
        st.markdown(f"**{class_id}.** {name}")

# ── Encabezado ───────────────────────────────────────────────────────────────
st.title("Detección de daños en autos")
st.caption("Modelo Faster R-CNN entrenado sobre CarDD · Subí una foto de un auto para detectar daños")

# ── Carga del modelo ─────────────────────────────────────────────────────────
with st.spinner("Cargando modelo..."):
    model = load_model(CHECKPOINT_PATH)

# ── Entrada de imagen ────────────────────────────────────────────────────────
st.subheader("Imagen de entrada")
input_tab, camera_tab = st.tabs(["Subir archivo", "Usar cámara"])

pil_image = None

with input_tab:
    uploaded = st.file_uploader(
        "Seleccioná una imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        pil_image = Image.open(uploaded)

with camera_tab:
    camera_photo = st.camera_input("Tomá una foto")
    if camera_photo is not None and pil_image is None:
        pil_image = Image.open(camera_photo)

# ── Inferencia y resultado ────────────────────────────────────────────────────
if pil_image is not None:
    st.divider()

    detections = run_inference(model, pil_image, score_threshold=score_threshold)
    result_image = draw_predictions(pil_image, detections)

    col_orig, col_result = st.columns(2)

    with col_orig:
        st.subheader("Imagen original")
        st.image(pil_image, use_container_width=True)

    with col_result:
        st.subheader(f"Detecciones (umbral {score_threshold:.0%})")
        st.image(result_image, use_container_width=True)

    st.divider()

    if detections:
        st.subheader(f"{len(detections)} daño(s) detectado(s)")

        rows = []
        for det in detections:
            x0, y0, x1, y1 = det["box"]
            area_px = (x1 - x0) * (y1 - y0)
            img_area = pil_image.width * pil_image.height
            rows.append({
                "Clase": det["class_name"],
                "Confianza": f"{det['score']:.1%}",
                "Área (px²)": f"{area_px:,}",
                "Área (%)": f"{area_px / img_area * 100:.1f}%",
                "Coordenadas [x0,y0,x1,y1]": str(det["box"]),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        buf = BytesIO()
        result_image.save(buf, format="PNG")
        st.download_button(
            "Descargar imagen con detecciones",
            data=buf.getvalue(),
            file_name="detecciones.png",
            mime="image/png",
        )
    else:
        st.info(
            f"No se detectaron daños con un umbral de confianza de {score_threshold:.0%}. "
            "Probá bajando el umbral en la barra lateral o usá otra imagen."
        )
else:
    st.info("Subí una imagen de un auto para comenzar.")
