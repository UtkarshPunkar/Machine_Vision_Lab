import cv2
import numpy as np
import time

# ============================================================
# FILE PATHS
# ============================================================

MODEL_CONFIG = r"C:\Users\utkar\Desktop\Package_Detection\yolov4-tiny.cfg"
MODEL_WEIGHTS = r"C:\Users\utkar\Desktop\Package_Detection\yolov4-tiny.weights"
CLASS_FILE = r"C:\Users\utkar\Desktop\Package_Detection\coco.names"

# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_FILE, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("Number of classes:", len(classes))

# ============================================================
# LOAD YOLO MODEL USING OPENCV DNN
# ============================================================

net = cv2.dnn.readNetFromDarknet(
    MODEL_CONFIG,
    MODEL_WEIGHTS
)

# CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ============================================================
# GET YOLO OUTPUT LAYERS
# ============================================================

layer_names = net.getLayerNames()

output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers().flatten()
]

print("YOLO model loaded successfully.")

# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press Q to exit.")

# ============================================================
# CONFIDENCE SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.40

# ============================================================
# FPS VARIABLES
# ============================================================

prev_time = 0

# ============================================================
# REAL-TIME DETECTION
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read camera frame.")
        break

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    frame = cv2.resize(frame, (800, 600))

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Create Blob
    # --------------------------------------------------------

    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    # --------------------------------------------------------
    # YOLO Forward Pass
    # --------------------------------------------------------

    net.setInput(blob)

    outputs = net.forward(output_layers)

    # --------------------------------------------------------
    # Store detections
    # --------------------------------------------------------

    boxes = []
    confidences = []
    class_ids = []

    # --------------------------------------------------------
    # Process YOLO Output
    # --------------------------------------------------------

    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # --------------------------------------------------------
    # Non-Maximum Suppression
    # --------------------------------------------------------

    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )

    detected_count = 0

    # --------------------------------------------------------
    # Draw Bounding Boxes
    # --------------------------------------------------------

    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            label = classes[class_ids[i]]

            confidence = confidences[i]

            # Draw rectangle

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Center point

            center_x = x + w // 2
            center_y = y + h // 2

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Label

            text = f"{label}: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detected_count += 1

    # ========================================================
    # FPS CALCULATION
    # ========================================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"Objects Detected: {detected_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # ========================================================
    # SHOW OUTPUT
    # ========================================================

    cv2.imshow(
        "YOLO Package/Object Detection",
        frame
    )

    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# RELEASE
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Program stopped.")