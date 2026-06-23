import cv2
import os

# Get current script folder
script_dir = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# 1. IMAGE LOADING AND DISPLAY
# =====================================================

print("\n===== IMAGE LOADING AND DISPLAY =====")

image_path = os.path.join(script_dir, "sample.jpg")

img = cv2.imread(image_path)

if img is None:
    print("sample.jpg not found.")
else:
    print("Image loaded successfully.")

    cv2.imshow("Original Image", img)

    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input("Press Enter for Image Properties...")

# =====================================================
# 2. IMAGE PROPERTIES
# =====================================================

print("\n===== IMAGE PROPERTIES =====")

if img is None:
    print("Image not available.")
else:
    print("Shape:", img.shape)
    print("Height:", img.shape[0])
    print("Width:", img.shape[1])
    print("Channels:", img.shape[2])
    print("Data Type:", img.dtype)
    print("Size:", img.size)

input("Press Enter for Color Channels...")

# =====================================================
# 3. COLOR CHANNELS
# =====================================================

print("\n===== COLOR CHANNELS =====")

if img is None:
    print("Image not available.")
else:
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    cv2.imshow("Blue Channel", blue)
    cv2.imshow("Green Channel", green)
    cv2.imshow("Red Channel", red)

    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input("Press Enter for Video Capture...")

# =====================================================
# 4. VIDEO CAPTURE FROM FILE
# =====================================================

print("\n===== VIDEO CAPTURE FROM FILE =====")

video_path = os.path.join(script_dir, "sample.mp4")

if not os.path.exists(video_path):
    print("sample.mp4 not found.")
else:
    cap = cv2.VideoCapture(video_path)

    print("Press Q to stop video.")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        cv2.imshow("Video Playback", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

input("Press Enter for Webcam Access...")

# =====================================================
# 5. WEBCAM ACCESS
# =====================================================

print("\n===== WEBCAM ACCESS =====")

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Unable to access webcam.")
else:

    print("Press Q to exit webcam.")

    while True:

        ret, frame = webcam.read()

        if not ret:
            break

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

print("\n===== PRACTICAL COMPLETED SUCCESSFULLY =====")
