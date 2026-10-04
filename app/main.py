"""FastAPI service for DeepFace video training and detection."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .services.deepface_service import (
    DetectedFace,
    deduplicate_embeddings,
    extract_faces,
    find_best_match,
    prepare_known_embeddings,
)
from .services.face_store import FaceStore
from .services.person_detector import PersonDetector
from .services.video_processing import FFmpegError, extract_frames, extract_frames_with_person_filter

_LOCAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR = os.getenv("AGENTS_DATA_DIR", str(_LOCAL_DATA_DIR))
TMP_DIR = os.getenv("AGENTS_TMP_DIR", "/tmp/agents")
MODEL_NAME = os.getenv("DEEPFACE_MODEL_NAME", "Facenet512")
DETECTOR_BACKEND = os.getenv("DEEPFACE_DETECTOR_BACKEND", "retinaface")
DEFAULT_FPS = float(os.getenv("VIDEO_SAMPLE_FPS", "1.0"))
DEFAULT_MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "85.0"))
MIN_UNKNOWN_CONFIDENCE = float(os.getenv("MIN_UNKNOWN_CONFIDENCE", "60.0"))
# Face quality gates (detector confidence 0-1, face size in px, Laplacian variance).
MIN_FACE_CONFIDENCE = float(os.getenv("MIN_FACE_CONFIDENCE", "0.9"))
TRAIN_MIN_FACE_SIZE = int(os.getenv("TRAIN_MIN_FACE_SIZE", "60"))
DETECT_MIN_FACE_SIZE = int(os.getenv("DETECT_MIN_FACE_SIZE", "30"))
TRAIN_MIN_SHARPNESS = float(os.getenv("TRAIN_MIN_SHARPNESS", "25.0"))
MATCH_TOP_K = int(os.getenv("MATCH_TOP_K", "3"))

FACE_DB_DIR = str(Path(DATA_DIR) / "faces")
Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title="blinkRoutines Agents", version="1.0.0")
store = FaceStore(FACE_DB_DIR)
person_detector = PersonDetector(confidence_threshold=0.5)


def _read_upload(video: UploadFile) -> tuple[bytes, str]:
    content = video.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Video file is empty")
    suffix = Path(video.filename or "input.mp4").suffix or ".mp4"
    return content, suffix


def _read_image_upload(image: UploadFile) -> tuple[bytes, str]:
    content = image.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Uploaded file must be an image")
    suffix = Path(image.filename or "input.jpg").suffix or ".jpg"
    return content, suffix


def _frame_time_seconds(frame_name: str, fps: float) -> float:
    """Timestamp of an extracted frame (ffmpeg numbers frames from 1)."""
    try:
        index = int(Path(frame_name).stem.rsplit("_", 1)[-1]) - 1
    except ValueError:
        return 0.0
    return round(max(0, index) / fps, 2)


def _detect_faces(image_path: str, training: bool) -> list[DetectedFace]:
    try:
        return extract_faces(
            image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            min_face_confidence=MIN_FACE_CONFIDENCE,
            min_face_size=TRAIN_MIN_FACE_SIZE if training else DETECT_MIN_FACE_SIZE,
            min_sharpness=TRAIN_MIN_SHARPNESS if training else 0.0,
        )
    except Exception as exc:  # pragma: no cover - runtime model/backend failures
        raise HTTPException(
            status_code=500,
            detail=f"DeepFace processing failed: {exc}",
        ) from exc


def _extract_faces_from_video_bytes(
    video_bytes: bytes,
    suffix: str,
    fps: float,
    training: bool,
    filter_empty_frames: bool = False,
) -> tuple[list[tuple[str, list[DetectedFace]]], int, int]:
    """Extract reliable faces from every sampled frame of a video.

    Args:
        video_bytes: Video file content
        suffix: File extension (e.g., ".mp4")
        fps: Frames per second to extract
        training: Apply the stricter training quality gates
        filter_empty_frames: Pre-filter frames with PersonDetector before DeepFace

    Returns:
        Tuple of (frames, frames_filtered_out, frames_total) where frames is a
        list of (frame_name, faces) for frames with at least one reliable face.
    """
    with tempfile.TemporaryDirectory(dir=TMP_DIR) as temp_dir:
        video_path = Path(temp_dir) / f"input{suffix}"
        video_path.write_bytes(video_bytes)

        frame_dir = Path(temp_dir) / "frames"
        try:
            if filter_empty_frames:
                frame_paths, filtered_out = extract_frames_with_person_filter(
                    str(video_path),
                    fps=fps,
                    output_dir=str(frame_dir),
                    person_detector=person_detector,
                )
            else:
                frame_paths = extract_frames(str(video_path), fps=fps, output_dir=str(frame_dir))
                filtered_out = 0
        except FFmpegError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        frames: list[tuple[str, list[DetectedFace]]] = []
        for frame_path in frame_paths:
            faces = _detect_faces(frame_path, training=training)
            if faces:
                frames.append((Path(frame_path).name, faces))
        return frames, filtered_out, len(frame_paths) + filtered_out


def _detect_faces_in_image_bytes(
    image_bytes: bytes,
    suffix: str,
    training: bool,
) -> list[DetectedFace]:
    with tempfile.TemporaryDirectory(dir=TMP_DIR) as temp_dir:
        image_path = Path(temp_dir) / f"input{suffix}"
        image_path.write_bytes(image_bytes)
        return _detect_faces(str(image_path), training=training)


def _match_face(face: DetectedFace, known: dict, min_confidence: float):
    return find_best_match(
        embedding=face.embedding,
        known_embeddings=known,
        min_confidence=min_confidence,
        min_unknown_confidence=MIN_UNKNOWN_CONFIDENCE,
        top_k=MATCH_TOP_K,
    )


def _detection_entry(frame: str, frame_time: float, face: DetectedFace, match) -> dict:
    return {
        "frame": frame,
        "frame_time_seconds": frame_time,
        "person_name": match.person_name,
        "confidence": round(match.confidence, 2),
        "distance": round(match.distance, 6),
        "face_confidence": round(face.face_confidence, 4),
        "facial_area": {key: face.area.get(key) for key in ("x", "y", "w", "h")},
    }


@app.get("/health")
def health() -> dict:
    """Health endpoint for docker-compose and probes."""
    return {
        "ok": True,
        "model": MODEL_NAME,
        "detector_backend": DETECTOR_BACKEND,
        "faces_db": FACE_DB_DIR,
    }


@app.get("/api/v2/list_faces")
def list_faces() -> dict:
    """List known people currently stored in the local face database."""
    people = store.list_people()
    return {"status_code": 200, "is_ok": True, "response": people}


@app.post("/api/v2/register_face/{person_name}")
def register_face(
    person_name: str,
    video: UploadFile = File(...),
    sample_fps: float = Form(DEFAULT_FPS),
    filter_empty_frames: bool = Form(False),
) -> dict:
    """Train/register a person using frames extracted from an uploaded video.

    Args:
        person_name: Name of the person to register
        video: Video file (MP4, AVI, etc.)
        sample_fps: Frames per second to extract
        filter_empty_frames: If True, pre-filter frames with PersonDetector. Not needed
            for accuracy: every frame already goes through the face quality gates.

    Only frames with exactly one good-quality face are used, so another person
    appearing in the video can never be stored under ``person_name``.

    Returns:
        Response with person_name, frames_used, frames_filtered_out, and total_embeddings.
    """
    if sample_fps <= 0:
        raise HTTPException(status_code=400, detail="sample_fps must be > 0")

    video_bytes, suffix = _read_upload(video)
    frames, frames_filtered, frames_total = _extract_faces_from_video_bytes(
        video_bytes,
        suffix,
        sample_fps,
        training=True,
        filter_empty_frames=filter_empty_frames,
    )

    single_face_frames = [faces[0].embedding for _, faces in frames if len(faces) == 1]
    vectors = deduplicate_embeddings(single_face_frames)
    if not vectors:
        raise HTTPException(
            status_code=422,
            detail=(
                "No usable faces could be extracted from provided video "
                f"({frames_total} frames sampled, {len(frames)} with a good-quality face, "
                f"{len(single_face_frames)} with exactly one face)"
            ),
        )

    total = store.append_embeddings(person_name, vectors)
    return {
        "status_code": 200,
        "is_ok": True,
        "response": {
            "person_name": person_name,
            "frames_used": len(vectors),
            "frames_filtered_out": frames_filtered,
            "frames_multiple_faces": len(frames) - len(single_face_frames),
            "frames_duplicated": len(single_face_frames) - len(vectors),
            "total_embeddings": total,
        },
    }


@app.post("/api/v2/detect_person")
def detect_person(
    video: UploadFile = File(...),
    sample_fps: float = Form(DEFAULT_FPS),
    min_confidence: float = Form(DEFAULT_MIN_CONFIDENCE),
) -> dict:
    """Detect known people in an uploaded video using stored embeddings."""
    if sample_fps <= 0:
        raise HTTPException(status_code=400, detail="sample_fps must be > 0")
    if min_confidence < 0 or min_confidence > 100:
        raise HTTPException(status_code=400, detail="min_confidence must be between 0 and 100")

    known = prepare_known_embeddings(store.load_all())
    if not known:
        raise HTTPException(status_code=404, detail="No registered faces available")

    video_bytes, suffix = _read_upload(video)
    frames, _, frames_total = _extract_faces_from_video_bytes(
        video_bytes,
        suffix,
        sample_fps,
        training=False,
    )

    detections = []
    summary: dict[str, dict] = {}
    for frame_name, faces in frames:
        frame_time = _frame_time_seconds(frame_name, sample_fps)
        for face in faces:
            match = _match_face(face, known, min_confidence)
            if not match:
                continue
            detections.append(_detection_entry(frame_name, frame_time, face, match))
            person = summary.setdefault(
                match.person_name,
                {"frames": 0, "max_confidence": 0.0, "first_seen_seconds": frame_time},
            )
            person["frames"] += 1
            person["max_confidence"] = max(person["max_confidence"], round(match.confidence, 2))
            person["last_seen_seconds"] = frame_time

    return {
        "status_code": 200,
        "is_ok": True,
        "response": {
            "frames_analyzed": frames_total,
            "frames_with_faces": len(frames),
            "detections": detections,
            "summary": summary,
        },
    }


@app.post("/api/v2/register_face_image/{person_name}")
def register_face_image(
    person_name: str,
    image: UploadFile = File(...),
) -> dict:
    """Train/register a person using a single uploaded image."""
    image_bytes, suffix = _read_image_upload(image)
    faces = _detect_faces_in_image_bytes(image_bytes, suffix, training=True)
    if not faces:
        raise HTTPException(
            status_code=422,
            detail="No good-quality face could be extracted from provided image",
        )
    if len(faces) > 1:
        raise HTTPException(
            status_code=422,
            detail=f"Image contains {len(faces)} faces; provide an image with only one person",
        )

    total = store.append_embeddings(person_name, [faces[0].embedding])
    return {
        "status_code": 200,
        "is_ok": True,
        "response": {
            "person_name": person_name,
            "frames_used": 1,
            "total_embeddings": total,
        },
    }


@app.post("/api/v2/detect_person_image")
def detect_person_image(
    image: UploadFile = File(...),
    min_confidence: float = Form(DEFAULT_MIN_CONFIDENCE),
) -> dict:
    """Detect known people in a single uploaded image."""
    if min_confidence < 0 or min_confidence > 100:
        raise HTTPException(status_code=400, detail="min_confidence must be between 0 and 100")

    known = prepare_known_embeddings(store.load_all())
    if not known:
        raise HTTPException(status_code=404, detail="No registered faces available")

    image_bytes, suffix = _read_image_upload(image)
    faces = _detect_faces_in_image_bytes(image_bytes, suffix, training=False)

    detections = []
    for face in faces:
        match = _match_face(face, known, min_confidence)
        if match:
            detections.append(_detection_entry("image", 0.0, face, match))

    return {
        "status_code": 200,
        "is_ok": True,
        "response": {
            "frames_analyzed": 1,
            "detections": detections,
        },
    }
