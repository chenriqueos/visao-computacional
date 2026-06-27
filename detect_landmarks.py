"""Detect facial landmarks (expression lines) on an image using MediaPipe."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_mp_face_mesh = mp.solutions.face_mesh
_mp_drawing = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


class LandmarkError(RuntimeError):
    """Raised when the landmark detection pipeline fails."""


def annotate_landmarks(
    image: np.ndarray,
    *,
    max_faces: int = 1,
    min_detection_confidence: float = 0.5,
    draw_tesselation: bool = False,
) -> tuple[np.ndarray, int]:
    """Return a copy of ``image`` with facial expression landmarks drawn.

    The contours drawn (eyes, eyebrows, lips, face oval and nose) correspond
    to the regions that define facial expression lines.
    """
    if image is None or image.size == 0:
        raise LandmarkError("Empty image provided to landmark detector")

    annotated = image.copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    faces = results.multi_face_landmarks or []
    for face_landmarks in faces:
        if draw_tesselation:
            _mp_drawing.draw_landmarks(
                image=annotated,
                landmark_list=face_landmarks,
                connections=_mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=_mp_styles.get_default_face_mesh_tesselation_style(),
            )
        _mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=_mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=_mp_styles.get_default_face_mesh_contours_style(),
        )

    return annotated, len(faces)


def annotate_file(
    input_path: Path,
    output_path: Path,
    *,
    max_faces: int = 1,
    min_detection_confidence: float = 0.5,
    draw_tesselation: bool = False,
) -> int:
    """Load ``input_path``, annotate facial landmarks and save to ``output_path``.

    Returns the number of faces detected.
    """
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image from {input_path}")

    annotated, n_faces = annotate_landmarks(
        image,
        max_faces=max_faces,
        min_detection_confidence=min_detection_confidence,
        draw_tesselation=draw_tesselation,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise LandmarkError(f"Failed to write annotated image to {output_path}")

    return n_faces


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_landmarks{input_path.suffix}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image path.")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (default: <input>_landmarks<ext>).",
    )
    parser.add_argument(
        "--max-faces", type=int, default=1,
        help="Maximum number of faces to detect (default: %(default)s).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum detection confidence in [0, 1] (default: %(default)s).",
    )
    parser.add_argument(
        "--tesselation", action="store_true",
        help="Also draw the full face mesh tesselation.",
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

    output = args.output or _default_output(args.input)
    try:
        n_faces = annotate_file(
            args.input,
            output,
            max_faces=args.max_faces,
            min_detection_confidence=args.confidence,
            draw_tesselation=args.tesselation,
        )
    except (FileNotFoundError, LandmarkError) as exc:
        logger.error("%s", exc)
        return 1

    if n_faces == 0:
        logger.warning("No face detected in %s", args.input)
    else:
        logger.info("Detected %d face(s); annotated image saved to %s", n_faces, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
