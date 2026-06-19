# `prod/utils.py`

## Proposito del archivo

`prod/utils.py` concentra la logica auxiliar de la app Streamlit final.

## Responsabilidades

- Resolver el checkpoint final desde `dev/modelo.pth` local o desde Google Drive.
- Cachear la carga del modelo con `@st.cache_resource`.
- Reconstruir `Faster R-CNN MobileNet V3 Large FPN` con 6 clases de dano + background.
- Aplicar el mismo preprocesamiento usado en test: imagen RGB a tensor, sin resize fijo ni normalizacion extra.
- Ejecutar inferencia estandar y modo `High-detail scan` por tiles.
- Aplicar NMS sobre detecciones tiled.
- Dibujar bounding boxes, labels y severidad sobre la imagen.
- Calcular severidad, rango de costo orientativo, cobertura simulada y tablas de presentacion.
- Cargar `dev/best_test_result.json` para mostrar metricas reales.

## Variables de entorno para despliegue

La app busca primero un checkpoint local en `dev/modelo.pth`. Si no existe, descarga desde Google Drive usando:

- `MODEL_GDRIVE_ID`
- `MODEL_GDRIVE_URL`

En Streamlit Cloud estas claves deben configurarse como secrets o variables del entorno.

## Nota tecnica

El checkpoint se carga con `torch.load(..., weights_only=False)` porque fue guardado como payload completo de entrenamiento, no como un `state_dict` aislado.
