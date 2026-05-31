"""Capture a single image from a connected camera using OpenCV."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import cv2

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("captured_image.jpg")
DEFAULT_CAMERA_INDEX = 0
DEFAULT_WARMUP_FRAMES = 5


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read from."""


@contextmanager
def open_camera(index: int) -> Iterator[cv2.VideoCapture]:
    """Yield an opened ``cv2.VideoCapture`` and release it on exit."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise CameraError(f"Could not open camera at index {index}")
        yield cap
    finally:
        cap.release()


def capture_image(
    output_path: Path = DEFAULT_OUTPUT,
    camera_index: int = DEFAULT_CAMERA_INDEX,
    warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    annotate: bool = False,
) -> Path:
    """Capture a frame from the camera and write it to ``output_path``.

    When ``annotate`` is true, facial expression landmarks are drawn on the
    captured frame before saving.
    """
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_camera(camera_index) as cap:
        # Discard the first frames so the sensor can adjust exposure/white-balance.
        for _ in range(warmup_frames):
            cap.read()

        ok, frame = cap.read()
        if not ok or frame is None:
            raise CameraError("Failed to capture image from camera")

    if annotate:
        from detect_landmarks import annotate_landmarks

        frame, n_faces = annotate_landmarks(frame)
        if n_faces == 0:
            logger.warning("No face detected in captured frame")
        else:
            logger.info("Annotated %d face(s) on captured frame", n_faces)

    if not cv2.imwrite(str(output_path), frame):
        raise CameraError(f"Failed to write image to {output_path}")

    logger.info("Image saved to %s", output_path)
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path where the captured image will be saved (default: %(default)s).",
    )
    parser.add_argument(
        "-c", "--camera", type=int, default=DEFAULT_CAMERA_INDEX,
        help="Camera device index (default: %(default)s).",
    )
    parser.add_argument(
        "-w", "--warmup", type=int, default=DEFAULT_WARMUP_FRAMES,
        help="Number of warmup frames to discard before capture (default: %(default)s).",
    )
    parser.add_argument(
        "-a", "--annotate", action="store_true",
        help="Draw facial expression landmarks on the captured image.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        capture_image(
            output_path=args.output,
            camera_index=args.camera,
            warmup_frames=args.warmup,
            annotate=args.annotate,
        )
    except (CameraError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
