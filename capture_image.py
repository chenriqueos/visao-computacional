import cv2

def capture_image(output_path: str = "captured_image.jpg") -> None:
    """Captures a single image from the default camera and saves it.

    Parameters
    ----------
    output_path: str
        Path where the captured image will be saved.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture image")

    cv2.imwrite(output_path, frame)
    cap.release()


if __name__ == "__main__":
    capture_image()
