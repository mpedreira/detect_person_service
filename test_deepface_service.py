"""Unit tests for face filtering and matching (DeepFace is mocked)."""

import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("AGENTS_DATA_DIR", tempfile.mkdtemp())
os.environ.setdefault("AGENTS_TMP_DIR", tempfile.mkdtemp())

from app.services import deepface_service  # noqa: E402
from app.services.deepface_service import (  # noqa: E402
    DetectedFace,
    deduplicate_embeddings,
    extract_faces,
    find_best_match,
    prepare_known_embeddings,
)


def _unit(seed: int, dim: int = 8) -> list[float]:
    vector = np.random.default_rng(seed).normal(size=dim)
    return list(vector / np.linalg.norm(vector))


def _fake_deepface(monkeypatch, representations):
    module = types.ModuleType("deepface")

    class FakeDeepFace:
        @staticmethod
        def represent(**_kwargs):
            if isinstance(representations, Exception):
                raise representations
            return representations

    module.DeepFace = FakeDeepFace
    monkeypatch.setitem(sys.modules, "deepface", module)


def _rep(confidence, w=100, h=100, seed=1):
    return {
        "embedding": _unit(seed),
        "face_confidence": confidence,
        "facial_area": {"x": 0, "y": 0, "w": w, "h": h},
    }


def test_extract_faces_discards_whole_image_fallback(monkeypatch):
    # No face detected: DeepFace returns the whole frame with confidence 0.
    _fake_deepface(monkeypatch, [_rep(0.0, w=1920, h=1080)])
    assert extract_faces("img.jpg", "Facenet512", "retinaface") == []


def test_extract_faces_returns_all_faces_largest_first(monkeypatch):
    _fake_deepface(monkeypatch, [_rep(0.99, w=50, h=50, seed=1), _rep(0.98, w=120, h=130, seed=2)])
    faces = extract_faces("img.jpg", "Facenet512", "retinaface")
    assert [face.size for face in faces] == [120, 50]


def test_extract_faces_filters_small_faces(monkeypatch):
    _fake_deepface(monkeypatch, [_rep(0.99, w=20, h=20), _rep(0.99, w=80, h=80)])
    faces = extract_faces("img.jpg", "Facenet512", "retinaface", min_face_size=40)
    assert len(faces) == 1 and faces[0].size == 80


def test_extract_faces_skip_backend_ignores_confidence(monkeypatch):
    _fake_deepface(monkeypatch, [_rep(0.0)])
    assert len(extract_faces("img.jpg", "Facenet512", "skip")) == 1


def test_extract_faces_no_face_error_returns_empty(monkeypatch):
    _fake_deepface(monkeypatch, ValueError("Face could not be detected"))
    assert extract_faces("img.jpg", "Facenet512", "retinaface") == []


def test_extract_faces_filters_blurry_faces(monkeypatch):
    import cv2

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        sharp = str(Path(tmp) / "sharp.png")
        blurry = str(Path(tmp) / "blurry.png")
        noise = rng.integers(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(sharp, noise)
        cv2.imwrite(blurry, cv2.GaussianBlur(noise, (31, 31), 10))
        _fake_deepface(monkeypatch, [_rep(0.99)])
        assert len(extract_faces(sharp, "m", "retinaface", min_sharpness=25)) == 1
        assert extract_faces(blurry, "m", "retinaface", min_sharpness=25) == []


def test_find_best_match_known_unknown_and_none():
    alice = _unit(1)
    known = prepare_known_embeddings({"alice": [alice], "bob": [_unit(2)]})

    match = find_best_match(alice, known, min_confidence=85, min_unknown_confidence=60)
    assert match.person_name == "alice" and match.confidence == pytest.approx(100, abs=1e-3)

    # Orthogonal-ish vector: very low similarity -> no detection.
    assert find_best_match(_unit(99), known, min_confidence=85, min_unknown_confidence=60) is None

    partial = np.asarray(alice) * 0.7 + np.asarray(_unit(3)) * 0.3
    match = find_best_match(list(partial), known, min_confidence=99, min_unknown_confidence=10)
    assert match.person_name == "unknown"


def test_find_best_match_single_outlier_does_not_win():
    query = _unit(10)
    # bob has one wrong embedding identical to the query and several unrelated ones.
    known = {
        "alice": [list(np.asarray(query) * 0.9 + np.asarray(_unit(s)) * 0.1) for s in (11, 12, 13)],
        "bob": [query, _unit(20), _unit(21), _unit(22)],
    }
    match = find_best_match(query, known, min_confidence=50, top_k=3)
    assert match.person_name == "alice"
    # With nearest-neighbour matching (top_k=1) the outlier would win.
    assert find_best_match(query, known, min_confidence=50, top_k=1).person_name == "bob"


def test_find_best_match_ignores_dimension_mismatch():
    known = {"alice": [_unit(1, dim=4)]}
    assert find_best_match(_unit(1), known, min_confidence=0) is None


def test_deduplicate_embeddings():
    base = _unit(1)
    near = list(np.asarray(base) + 1e-4)
    assert len(deduplicate_embeddings([base, near, _unit(2)])) == 2


def test_frame_time_uses_frame_number():
    from app import main

    assert main._frame_time_seconds("frame_000001.jpg", 2.0) == 0.0
    assert main._frame_time_seconds("frame_000007.jpg", 2.0) == 3.0


def test_register_video_skips_multi_face_frames(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    face_a = DetectedFace(embedding=_unit(1), face_confidence=0.99, area={"w": 100, "h": 100})
    face_b = DetectedFace(embedding=_unit(2), face_confidence=0.99, area={"w": 90, "h": 90})
    frames = [
        ("frame_000001.jpg", [face_a]),
        ("frame_000002.jpg", [face_a, face_b]),
        ("frame_000003.jpg", [face_a]),
    ]
    monkeypatch.setattr(main, "_extract_faces_from_video_bytes", lambda *a, **k: (frames, 0, 4))
    stored = {}
    monkeypatch.setattr(
        main.store, "append_embeddings", lambda name, vectors: stored.setdefault(name, vectors) and len(vectors)
    )

    response = TestClient(main.app).post(
        "/api/v2/register_face/alice", files={"video": ("v.mp4", b"data", "video/mp4")}
    )
    body = response.json()["response"]
    assert response.status_code == 200
    assert body["frames_used"] == 1  # frame 3 is a duplicate of frame 1
    assert body["frames_multiple_faces"] == 1
    assert body["frames_duplicated"] == 1
    assert stored["alice"] == [face_a.embedding]


def test_detect_video_reports_every_face_with_timestamps(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    alice, bob = _unit(1), _unit(2)
    monkeypatch.setattr(main.store, "load_all", lambda: {"alice": [alice], "bob": [bob]})
    frames = [
        ("frame_000003.jpg", [
            DetectedFace(embedding=alice, face_confidence=0.99, area={"w": 80, "h": 80}),
            DetectedFace(embedding=bob, face_confidence=0.99, area={"w": 60, "h": 60}),
        ]),
    ]
    monkeypatch.setattr(main, "_extract_faces_from_video_bytes", lambda *a, **k: (frames, 0, 5))

    response = TestClient(main.app).post(
        "/api/v2/detect_person",
        files={"video": ("v.mp4", b"data", "video/mp4")},
        data={"sample_fps": "1.0"},
    )
    body = response.json()["response"]
    assert response.status_code == 200
    assert body["frames_analyzed"] == 5
    assert [d["person_name"] for d in body["detections"]] == ["alice", "bob"]
    assert all(d["frame_time_seconds"] == 2.0 for d in body["detections"])
    assert body["summary"]["alice"]["frames"] == 1


def test_module_has_no_top_level_deepface_import():
    assert "DeepFace" not in vars(deepface_service)
