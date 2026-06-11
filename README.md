# blinkRoutines Agents

Servicio FastAPI para deteccion y entrenamiento facial desde video usando DeepFace + ffmpeg.

## Endpoints

- `GET /health`
- `GET /api/v2/list_faces`
- `POST /api/v2/register_face/{person_name}`
  - multipart/form-data: `video` (archivo)
  - opcional: `sample_fps` (float)
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
  - Si detecta rostro y coincide con persona registrada sobre el umbral, devuelve `person_name` conocido.
  - Si detecta rostro pero no alcanza umbral de coincidencia, devuelve `person_name: "unknown"`.
  - Si no detecta rostro, no agrega entrada en `detections`.

## Ejecutar con Docker

```bash
cd agents
docker compose up --build
```

Servicio disponible en `http://localhost:8010`.

## Persistencia

Los embeddings se guardan en `agents/data/faces/*.json` (volumen montado en `/data`).
