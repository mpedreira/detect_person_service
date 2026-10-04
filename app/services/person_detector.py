"""Person detection using MediaPipe with OpenCV fallback for filtering frames during training."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


class PersonDetector:
    """Detect human persons in an image using MediaPipe (with OpenCV fallback)."""

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize PersonDetector with optional confidence threshold.

        Args:
            confidence_threshold: Detection confidence threshold (0.0 to 1.0).
                                 Default 0.5 is conservative (accepts more frames).
        """
        self.confidence_threshold = confidence_threshold
        self._detector = None
        self._use_opencv = False

    def _ensure_loaded(self) -> None:
        """Lazy-load MediaPipe or OpenCV model on first use."""
        if self._detector is not None:
            return

        mediapipe_exc = None
        # Try MediaPipe first (preferred)
        try:
            import mediapipe as mp  # pylint: disable=import-outside-toplevel

            # mediapipe >= 0.10.30 / 1.x dropped the legacy "solutions" API.
            if not hasattr(mp, "solutions"):
                raise RuntimeError(
                    f"mediapipe {getattr(mp, '__version__', '?')} has no legacy solutions API"
                )
            self._detector = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=0,  # lite model for speed
                min_detection_confidence=self.confidence_threshold,
            )
            logger.info("PersonDetector: Using MediaPipe Pose")
            return
        except Exception as exc:
            mediapipe_exc = exc
            logger.debug("MediaPipe not available: %s", exc)

        # Fallback to OpenCV HAARCascade
        try:
            import cv2  # pylint: disable=import-outside-toplevel
            import os

            # Load pre-trained cascade classifier
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(cascade_path):
                raise RuntimeError(f"OpenCV cascade not found at {cascade_path}")

            self._detector = cv2.CascadeClassifier(cascade_path)
            if self._detector.empty():
                raise RuntimeError("Failed to load OpenCV cascade classifier")

            self._use_opencv = True
            logger.info("PersonDetector: Using OpenCV HAARCascade (fallback)")
            return
        except Exception as opencv_exc:
            logger.error("Failed to load both MediaPipe and OpenCV: %s", opencv_exc)
            raise RuntimeError(
                f"No person detection backend available. "
                f"MediaPipe: {mediapipe_exc}, OpenCV: {opencv_exc}"
            ) from opencv_exc

    def has_person(self, image_path: str) -> bool:
        """Check if an image contains a detected person.

        Args:
            image_path: Path to image file (JPEG, PNG, etc.)

        Returns:
            True if person detected above confidence threshold, False otherwise.
        """
        self._ensure_loaded()

        try:
            if self._use_opencv:
                return self._detect_with_opencv(image_path)
            else:
                return self._detect_with_mediapipe(image_path)
        except Exception as exc:
            logger.error("Error detecting person in %s: %s", image_path, exc)
            # Graceful degradation: assume no person on error to avoid filtering all frames
            return False

    def _detect_with_mediapipe(self, image_path: str) -> bool:
        """Detect person using MediaPipe Pose."""
        import numpy as np  # pylint: disable=import-outside-toplevel
        from PIL import Image  # pylint: disable=import-outside-toplevel

        try:
            # MediaPipe expects an RGB numpy array, not a PIL image.
            image = np.asarray(Image.open(image_path).convert("RGB"))
            result = self._detector.process(image)

            has_landmarks = (
                result.pose_landmarks is not None
                and len(result.pose_landmarks.landmark) > 0
            )
            if has_landmarks:
                logger.debug("Person detected in %s via MediaPipe", image_path)
                return True
            return False
        except Exception as exc:
            logger.warning("MediaPipe detection failed for %s: %s", image_path, exc)
            return False

    def _detect_with_opencv(self, image_path: str) -> bool:
        """Detect person/face using OpenCV HAARCascade."""
        import cv2  # pylint: disable=import-outside-toplevel

        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.warning("Failed to read image: %s", image_path)
                return False

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            if len(faces) > 0:
                logger.debug("Person/Face detected in %s via OpenCV (%d detections)", image_path, len(faces))
                return True
            return False
        except Exception as exc:
            logger.warning("OpenCV detection failed for %s: %s", image_path, exc)
            return False
