from ultralytics import YOLO
import cv2

def detector_YOLO(original_image, detector):
    # Make a copy and resize for YOLO
    resized_image = cv2.resize(original_image, (640, 640))
    h_orig, w_orig = original_image.shape[:2]
    h_resized, w_resized = 640, 640

    results = detector(resized_image)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if conf < 0.5:
            continue

        # Scale coordinates back to original image size
        scale_x = w_orig / w_resized
        scale_y = h_orig / h_resized
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        face = original_image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Optional: draw bounding box
        # cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return face  # Return the high-res face

    return None