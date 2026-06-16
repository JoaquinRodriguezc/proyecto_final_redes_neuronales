from __future__ import annotations

import base64
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

from app_utils import (
    CLASS_NAMES,
    CHECKPOINT_PATH,
    draw_predictions,
    load_model,
    run_inference,
    run_inference_two_pass,
)

# ── Perfil de daños: severidad, reparación y costos por clase ────────────────
# thresholds: (leve, moderado) como % del área total de la imagen
DAMAGE_PROFILE: dict[int, dict] = {
    1: {  # dent
        "thresholds": (0.03, 0.08),
        "repair": ["Sacabollos (PDR)", "Chapa y pintura", "Chapa pesada y repintado"],
        "costs": [
            {"Chapa": 0,  "Pintura": 20, "Mano de obra": 40},   # leve:     ~$60
            {"Chapa": 30, "Pintura": 50, "Mano de obra": 70},   # moderado: ~$150
            {"Chapa": 80, "Pintura": 100,"Mano de obra": 100},  # grave:    ~$280
        ],
        "insurance_covered": False,
    },
    2: {  # scratch
        "thresholds": (0.02, 0.06),
        "repair": ["Pulido y abrillantado", "Pintura parcial", "Pintura completa"],
        "costs": [
            {"Chapa": 0,  "Pintura": 20, "Mano de obra": 30},   # leve:     ~$50
            {"Chapa": 0,  "Pintura": 70, "Mano de obra": 60},   # moderado: ~$130
            {"Chapa": 30, "Pintura": 150,"Mano de obra": 100},  # grave:    ~$280
        ],
        "insurance_covered": False,
    },
    3: {  # crack
        "thresholds": (0.01, 0.04),
        "repair": ["Sellado de fisura", "Reparación de panel", "Chapa y repintado"],
        "costs": [
            {"Chapa": 0,  "Pintura": 20, "Mano de obra": 40},   # leve:     ~$60
            {"Chapa": 30, "Pintura": 60, "Mano de obra": 80},   # moderado: ~$170
            {"Chapa": 80, "Pintura": 120,"Mano de obra": 120},  # grave:    ~$320
        ],
        "insurance_covered": False,
    },
    4: {  # glass shatter
        "thresholds": (0.1, 0.3),
        "repair": ["Reemplazo de vidrio", "Reemplazo de vidrio", "Reemplazo de vidrio"],
        "costs": [
            {"Chapa": 0, "Pintura": 0, "Mano de obra": 200},
            {"Chapa": 0, "Pintura": 0, "Mano de obra": 500},
            {"Chapa": 0, "Pintura": 0, "Mano de obra": 900},
        ],
        "insurance_covered": True,
    },
    5: {  # lamp broken
        "thresholds": (0.02, 0.08),
        "repair": ["Reparación de faro", "Reemplazo de faro", "Reemplazo de conjunto"],
        "costs": [
            {"Chapa": 0,  "Pintura": 0, "Mano de obra": 80},
            {"Chapa": 0,  "Pintura": 0, "Mano de obra": 200},
            {"Chapa": 50, "Pintura": 50,"Mano de obra": 250},
        ],
        "insurance_covered": True,
    },
    6: {  # tire flat
        "thresholds": (0.05, 0.15),
        "repair": ["Reparacion / inflado", "Reemplazo de cubierta", "Reemplazo de cubierta y llanta"],
        "costs": [
            {"Chapa": 0, "Pintura": 0, "Mano de obra": 30},
            {"Chapa": 0, "Pintura": 0, "Mano de obra": 100},
            {"Chapa": 0, "Pintura": 0, "Mano de obra": 220},
        ],
        "insurance_covered": True,
    },
}

SEVERITY_LABELS = ["Leve", "Moderado", "Grave"]


def get_severity(area_pct: float, thresholds: tuple[float, float]) -> int:
    lo, hi = thresholds
    if area_pct < lo:
        return 0
    if area_pct < hi:
        return 1
    return 2


def generate_report_html(
    result_image: Image.Image,
    cost_rows: list[dict],
    total_min: int,
    total_max: int,
    insurance_recommendation: str,
) -> str:
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    rows_html = "".join(
        f"<tr><td>{r['Daño']}</td><td>{r['Severidad']}</td><td>{r['Reparación sugerida']}</td>"
        f"<td>${r['Chapa']}</td><td>${r['Pintura']}</td><td>${r['Mano de obra']}</td><td>${r['Total']}</td></tr>"
        for r in cost_rows
    )

    return f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"><title>Reporte de daños</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; color: #1a1a2e; }}
  h1 {{ color: #e94560; }} h2 {{ color: #16213e; border-bottom: 2px solid #e94560; padding-bottom: 6px; }}
  img {{ width: 100%; border-radius: 8px; margin: 16px 0; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ background: #16213e; color: white; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f4f4f4; }}
  .total {{ font-size: 1.2rem; font-weight: bold; margin: 12px 0; }}
  .recommendation {{ background: #eaf4fb; border-left: 4px solid #2980b9; padding: 12px 16px; border-radius: 4px; }}
  .footer {{ margin-top: 32px; font-size: 0.8rem; color: #888; }}
</style>
</head>
<body>
  <h1>Reporte de daños detectados</h1>
  <h2>Imagen analizada</h2>
  <img src="data:image/png;base64,{img_b64}" alt="Imagen con detecciones">
  <h2>Desglose de costos estimados</h2>
  <table>
    <tr><th>Daño</th><th>Severidad</th><th>Reparación</th><th>Chapa</th><th>Pintura</th><th>Mano de obra</th><th>Total</th></tr>
    {rows_html}
  </table>
  <p class="total">Total estimado: ${total_min:,} – ${total_max:,} USD</p>
  <h2>Recomendación de seguro</h2>
  <div class="recommendation">{insurance_recommendation}</div>
  <p class="footer">Valores orientativos en USD. No constituyen una cotización oficial.</p>
</body>
</html>"""


# ── Configuración de página ───────────────────────────────────────────────────
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

    use_two_pass = st.toggle(
        "Verificación en 2 pasadas",
        value=False,
        help="Primera pasada detecta candidatos con umbral bajo. Segunda pasada recorta y verifica cada uno. Reduce falsos positivos.",
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

    if use_two_pass:
        with st.spinner("Verificando en 2 pasadas..."):
            detections = run_inference_two_pass(model, pil_image, score_threshold=score_threshold)
    else:
        detections = run_inference(model, pil_image, score_threshold=score_threshold)

    result_image = draw_predictions(pil_image, detections)

    col_orig, col_result = st.columns(2)
    with col_orig:
        st.subheader("Imagen original")
        st.image(pil_image, use_container_width=True)
    with col_result:
        label = "2 pasadas" if use_two_pass else f"umbral {score_threshold:.0%}"
        st.subheader(f"Detecciones ({label})")
        st.image(result_image, use_container_width=True)

    st.divider()

    if not detections:
        st.info(
            f"No se detectaron daños con un umbral de confianza de {score_threshold:.0%}. "
            "Probá bajando el umbral en la barra lateral o usá otra imagen."
        )
    else:
        # ── Diagnóstico y desglose de costos ─────────────────────────────────
        st.subheader(f"{len(detections)} daño(s) detectado(s)")

        img_area = pil_image.width * pil_image.height
        cost_rows = []
        total_min, total_max = 0, 0

        for det in detections:
            x0, y0, x1, y1 = det["box"]
            area_pct = (x1 - x0) * (y1 - y0) / img_area
            profile = DAMAGE_PROFILE.get(det["label"], {})
            sev_idx = get_severity(area_pct, profile.get("thresholds", (0.05, 0.15)))
            costs = profile.get("costs", [{}])[sev_idx]
            repair = profile.get("repair", ["—"])[sev_idx]
            subtotal = sum(costs.values())
            total_min += subtotal
            total_max += int(subtotal * 1.4)
            cost_rows.append({
                "Daño": det["class_name"],
                "Severidad": SEVERITY_LABELS[sev_idx],
                "Reparación sugerida": repair,
                "Chapa": costs.get("Chapa", 0),
                "Pintura": costs.get("Pintura", 0),
                "Mano de obra": costs.get("Mano de obra", 0),
                "Total": subtotal,
            })

        st.dataframe(
            pd.DataFrame(cost_rows).rename(columns={
                "Chapa": "Chapa (USD)",
                "Pintura": "Pintura (USD)",
                "Mano de obra": "M. de obra (USD)",
                "Total": "Total (USD)",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(f"**Costo total estimado: ${total_min:,} – ${total_max:,} USD**")
        st.caption("Rango calculado sobre costo base (mín) y +40% por variación de taller (máx). Valores orientativos.")

        # ── Datos técnicos (colapsado) ────────────────────────────────────────
        with st.expander("Ver datos técnicos del análisis"):
            tech_rows = []
            for det in detections:
                x0, y0, x1, y1 = det["box"]
                area_px = (x1 - x0) * (y1 - y0)
                tech_rows.append({
                    "Clase": det["class_name"],
                    "Confianza": f"{det['score']:.1%}",
                    "Area (px2)": f"{area_px:,}",
                    "Area (%)": f"{area_px / img_area * 100:.1f}%",
                    "Coordenadas": str(det["box"]),
                })
            st.dataframe(pd.DataFrame(tech_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Calculadora de seguro ─────────────────────────────────────────────
        st.subheader("Calculadora de seguro")

        franquicia = st.number_input(
            "Tu franquicia / deducible (USD)",
            min_value=0,
            value=500,
            step=50,
            help="Ingresá el monto de tu franquicia para saber si conviene usar el seguro.",
        )

        costo_medio = (total_min + total_max) / 2
        covered_labels = {d["label"] for d in detections if DAMAGE_PROFILE.get(d["label"], {}).get("insurance_covered")}
        not_covered_labels = {d["label"] for d in detections} - covered_labels
        covered_names = [CLASS_NAMES[l] for l in covered_labels]
        not_covered_names = [CLASS_NAMES[l] for l in not_covered_labels]

        if costo_medio > franquicia:
            insurance_rec = (
                f"Conviene usar el seguro — el costo estimado (USD {costo_medio:,.0f}) "
                f"supera tu franquicia (USD {franquicia:,})."
            )
            st.success(insurance_rec)
        else:
            insurance_rec = (
                f"Conviene pagar particular — el costo estimado (USD {costo_medio:,.0f}) "
                f"es menor que tu franquicia (USD {franquicia:,})."
            )
            st.warning(insurance_rec)

        st.divider()

        # ── CTAs ──────────────────────────────────────────────────────────────
        cta1, cta2, cta3 = st.columns(3)

        with cta1:
            buf = BytesIO()
            result_image.save(buf, format="PNG")
            st.download_button(
                "Descargar imagen con detecciones",
                data=buf.getvalue(),
                file_name="detecciones.png",
                mime="image/png",
                use_container_width=True,
            )

        with cta2:
            report_html = generate_report_html(
                result_image, cost_rows, total_min, total_max, insurance_rec
            )
            st.download_button(
                "Descargar reporte (HTML)",
                data=report_html.encode("utf-8"),
                file_name="reporte_danos.html",
                mime="text/html",
                use_container_width=True,
            )

        with cta3:
            st.link_button(
                "Buscar talleres cercanos",
                "https://www.google.com/maps/search/taller+de+chapa+y+pintura",
                use_container_width=True,
            )

else:
    st.info("Subí una imagen de un auto para comenzar.")
