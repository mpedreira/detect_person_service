# blinkRoutines Agents

Servicio FastAPI para deteccion y entrenamiento facial desde video usando DeepFace + ffmpeg.

## Endpoints

- `GET /health`
- `GET /api/v2/list_faces`
- `POST /api/v2/register_face/{person_name}`
  - multipart/form-data: `video` (archivo)
  - opcional: `sample_fps` (float)
  - opcional: `filter_empty_frames` (bool, por defecto `false`; prefiltro con PersonDetector)
- `POST /api/v2/register_face_image/{person_name}`
  - multipart/form-data: `image` (archivo)
- `POST /api/v2/detect_person`
  - multipart/form-data: `video` (archivo)
  - opcional: `sample_fps` (float)
  - opcional: `min_confidence` (0-100)
- `POST /api/v2/detect_person_image`
  - multipart/form-data: `image` (archivo)
  - opcional: `min_confidence` (0-100)

## Comportamiento

- Entrenamiento (`/api/v2/register_face/{person_name}`, `/api/v2/register_face_image/{person_name}`):
  - Solo guarda embeddings si se detecta rostro.
  - Si no se detecta rostro, responde `422`.
- Deteccion (`/api/v2/detect_person`, `/api/v2/detect_person_image`):
  - Si detecta rostro y coincide con persona registrada >= 85% de confianza, devuelve `person_name` conocido.
  - Si detecta rostro con 60-85% de confianza, devuelve `person_name: "unknown"`.
  - Si detecta rostro pero confianza < 60%, no agrega entrada en `detections` (muy poco confiable).
  - Si no detecta rostro, no agrega entrada en `detections`.
  
  - Se analizan **todas** las caras de cada frame (no solo la primera).
  - En video, la respuesta incluye `summary` por persona (frames, confianza máxima, primera/última aparición).

- Calidad de cara (evita falsos positivos y embeddings malos):
  - Se descartan caras con confianza del detector < `MIN_FACE_CONFIDENCE`. Cuando DeepFace no
    encuentra cara devuelve el frame entero con confianza 0; ese "rostro" ya no se compara nunca.
  - Se descartan caras pequeñas (`TRAIN_MIN_FACE_SIZE` / `DETECT_MIN_FACE_SIZE`) y, en entrenamiento,
    caras borrosas (`TRAIN_MIN_SHARPNESS`, varianza del Laplaciano).
  - En entrenamiento por video solo se usan frames con **una única cara** y se eliminan embeddings
    casi duplicados. En entrenamiento por imagen, una imagen con varias caras responde `422`.
  - El matching puntúa cada persona con la media de sus `MATCH_TOP_K` embeddings más parecidos,
    así un único embedding erróneo no provoca un falso positivo.

  Variables de entorno:
  - `MIN_CONFIDENCE=85.0` (umbral para reportar persona conocida)
  - `MIN_UNKNOWN_CONFIDENCE=60.0` (umbral mínimo para reportar como unknown)
  - `MIN_FACE_CONFIDENCE=0.9` (confianza mínima del detector de caras, 0-1)
  - `TRAIN_MIN_FACE_SIZE=60` / `DETECT_MIN_FACE_SIZE=30` (lado mínimo de la cara en px)
  - `TRAIN_MIN_SHARPNESS=25.0` (nitidez mínima en entrenamiento; `0` desactiva)
  - `MATCH_TOP_K=3`

## Tests

```bash
pip install pytest httpx numpy opencv-python-headless fastapi python-multipart
pytest test_deepface_service.py
```

## Ejecutar con Docker

```bash
cd agents
docker compose up --build
```

Servicio disponible en `http://localhost:8010`.

## Persistencia

Los embeddings se guardan en `agents/data/faces/*.json` (volumen montado en `/data`).
