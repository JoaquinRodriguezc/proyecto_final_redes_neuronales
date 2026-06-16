from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from app_utils import CLASS_NAMES, CHECKPOINT_PATH, draw_predictions, load_model, run_inference

REPAIR_COST_RANGE: dict[int, tuple[int, int]] = {
    1: (200, 800),
    2: (100, 500),
    3: (50, 200),
    4: (300, 1500),
    5: (150, 600),
    6: (50, 200),
}

INSURANCE_COVERAGE: dict[str, set[int]] = {
    "Responsabilidad Civil": set(),
    "Terceros completo": {4, 5, 6},
    "Todo riesgo": {1, 2, 3, 4, 5, 6},
}

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

    if detections:
        st.divider()
        st.subheader("Estimación de costos y cobertura de seguro")
        st.caption("Valores orientativos en USD. No constituyen una cotización oficial.")

        detected_labels = {d["label"] for d in detections}

        cost_min = sum(REPAIR_COST_RANGE[d["label"]][0] for d in detections if d["label"] in REPAIR_COST_RANGE)
        cost_max = sum(REPAIR_COST_RANGE[d["label"]][1] for d in detections if d["label"] in REPAIR_COST_RANGE)

        col_cost, col_ins = st.columns(2)

        with col_cost:
            st.markdown("**Costo estimado de reparación**")
            cost_rows = []
            for det in detections:
                label = det["label"]
                if label in REPAIR_COST_RANGE:
                    lo, hi = REPAIR_COST_RANGE[label]
                    cost_rows.append({"Daño": det["class_name"], "Rango (USD)": f"${lo} – ${hi}"})
            st.dataframe(pd.DataFrame(cost_rows), use_container_width=True, hide_index=True)
            st.markdown(f"**Total estimado: ${cost_min:,} – ${cost_max:,} USD**")

        with col_ins:
            st.markdown("**Cobertura según tipo de seguro**")
            ins_rows = []
            for policy, covered_labels in INSURANCE_COVERAGE.items():
                covered = [CLASS_NAMES[l] for l in detected_labels if l in covered_labels]
                not_covered = [CLASS_NAMES[l] for l in detected_labels if l not in covered_labels]
                ins_rows.append({
                    "Seguro": policy,
                    "Cubre": ", ".join(covered) if covered else "Ninguno",
                    "No cubre": ", ".join(not_covered) if not_covered else "—",
                })
            st.dataframe(pd.DataFrame(ins_rows), use_container_width=True, hide_index=True)

else:
    st.info("Subí una imagen de un auto para comenzar.")
