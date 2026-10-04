"""DeepFace wrappers for face extraction, quality filtering and matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Detector backends that do not actually detect faces (no meaningful confidence).
_NON_DETECTING_BACKENDS = {"skip"}


@dataclass
class FaceMatch:
    """Single face match result."""

    person_name: str
    confidence: float
    distance: float


@dataclass
class DetectedFace:
    """A face found in an image together with its embedding and quality data."""

    embedding: list[float]
    face_confidence: float
    area: dict
    sharpness: float | None = None

    @property
    def size(self) -> int:
        """Smallest side of the face bounding box in pixels."""
        return int(min(self.area.get("w", 0), self.area.get("h", 0)))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    a = np.asarray(vec1, dtype=np.float32)
    b = np.asarray(vec2, dtype=np.float32)
    norm1 = float(np.linalg.norm(a))
    norm2 = float(np.linalg.norm(b))
    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    cosine_similarity = float(np.dot(a, b)) / (norm1 * norm2)
    return max(0.0, min(2.0, 1.0 - cosine_similarity))


def _face_sharpness(image_path: str, area: dict) -> float | None:
    """Variance of the Laplacian over the face crop (higher = sharper)."""
    import cv2  # pylint: disable=import-outside-toplevel

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    x, y = max(0, int(area.get("x", 0))), max(0, int(area.get("y", 0)))
    w, h = int(area.get("w", 0)), int(area.get("h", 0))
    crop = image[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def extract_faces(
    image_path: str,
    model_name: str,
    detector_backend: str,
    min_face_confidence: float = 0.9,
    min_face_size: int = 0,
    min_sharpness: float = 0.0,
) -> list[DetectedFace]:
    """Extract every reliable face in an image, largest first.

    Faces are discarded when the detector is not confident enough, when the
    face is too small to produce a useful embedding, or when it is blurry.
    When the detector finds nothing, DeepFace (with enforce_detection=False)
    returns the whole image with confidence 0; that "face" is discarded too,
    so it can never be matched against the registered people.
    """
    # Import lazily so module import remains lightweight until needed.
    from deepface import DeepFace  # pylint: disable=import-outside-toplevel

    try:
        representations = DeepFace.represent(
            img_path=image_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )
    except ValueError as exc:
        logger.debug("No face extracted from %s: %s", image_path, exc)
        return []

    check_confidence = detector_backend not in _NON_DETECTING_BACKENDS
    faces: list[DetectedFace] = []
    for representation in representations or []:
        embedding = representation.get("embedding")
        if not embedding:
            continue
        face = DetectedFace(
            embedding=[float(value) for value in embedding],
            face_confidence=float(representation.get("face_confidence") or 0.0),
            area=representation.get("facial_area") or {},
        )
        if check_confidence and face.face_confidence < min_face_confidence:
            continue
        if face.size < min_face_size:
            continue
        if min_sharpness > 0:
            face.sharpness = _face_sharpness(image_path, face.area)
            if face.sharpness is not None and face.sharpness < min_sharpness:
                continue
        faces.append(face)

    faces.sort(key=lambda item: item.size, reverse=True)
    return faces


def extract_embedding(
    image_path: str,
    model_name: str,
    detector_backend: str,
    enforce_detection: bool = False,
) -> list[float] | None:
    """Extract the embedding of the largest reliable face in an image."""
    faces = extract_faces(
        image_path,
        model_name=model_name,
        detector_backend=detector_backend,
        min_face_confidence=0.9 if enforce_detection else 0.0,
    )
    if not faces:
        return None
    return faces[0].embedding


def extract_embedding_with_person_filter(
    image_path: str,
    model_name: str,
    detector_backend: str,
    person_detector,
) -> list[float] | None:
    """Extract embedding only if a person is detected in the image."""
    if not person_detector.has_person(image_path):
        return None
    return extract_embedding(image_path, model_name, detector_backend)


def deduplicate_embeddings(
    embeddings: list[list[float]],
    max_similarity: float = 0.98,
) -> list[list[float]]:
    """Drop near-identical embeddings (e.g. consecutive frames of a still face).

    Duplicates add no information but bias the matching towards a single pose.
    """
    kept: list[list[float]] = []
    kept_normalized: list[np.ndarray] = []
    for embedding in embeddings:
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            continue
        vector = vector / norm
        if kept_normalized and float(np.max(np.stack(kept_normalized) @ vector)) >= max_similarity:
            continue
        kept.append(embedding)
        kept_normalized.append(vector)
    return kept


def prepare_known_embeddings(
    known_embeddings: dict[str, list[list[float]]],
) -> dict[str, np.ndarray]:
    """Convert stored embeddings into L2-normalized matrices (one per person)."""
    prepared: dict[str, np.ndarray] = {}
    for person_name, people_embeddings in known_embeddings.items():
        if not people_embeddings:
            continue
        matrix = np.asarray(people_embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            continue
        prepared[person_name] = _normalize_rows(matrix)
    return prepared


def find_best_match(
    embedding: list[float],
    known_embeddings: dict[str, list[list[float]]] | dict[str, np.ndarray],
    min_confidence: float,
    min_unknown_confidence: float = 0.0,
    top_k: int = 3,
) -> FaceMatch | None:
    """Find closest known person, or mark as unknown if confidence in range.

    Each person is scored with the mean of their ``top_k`` most similar
    registered embeddings instead of the single closest one, so one bad
    training frame (blurred, another person, etc.) cannot produce a false
    positive on its own.

    Returns:
        FaceMatch with known person if >= min_confidence,
        FaceMatch with "unknown" if min_unknown_confidence <= confidence < min_confidence,
        None if confidence < min_unknown_confidence or no embeddings to compare.
    """
    query = np.asarray(embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return None
    query = query / query_norm

    best: FaceMatch | None = None
    for person_name, people_embeddings in known_embeddings.items():
        matrix = np.asarray(people_embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != query.shape[0]:
            continue
        similarities = _normalize_rows(matrix) @ query
        k = max(1, min(top_k, similarities.shape[0]))
        score = float(np.mean(np.partition(similarities, -k)[-k:]))
        distance = max(0.0, min(2.0, 1.0 - score))
        if best is None or distance < best.distance:
            best = FaceMatch(
                person_name=person_name,
                confidence=max(0.0, min(100.0, score * 100.0)),
                distance=distance,
            )
    if best is None:
        return None
    if best.confidence >= min_confidence:
        return best
    if best.confidence >= min_unknown_confidence:
        return FaceMatch(
            person_name="unknown",
            confidence=best.confidence,
            distance=best.distance,
        )
    return None
