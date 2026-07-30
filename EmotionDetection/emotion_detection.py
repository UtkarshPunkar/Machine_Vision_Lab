from deepface import DeepFace
import cv2

import os
import cv2

img_path = r"C:\Users\utkar\Desktop\EmotionDetection\person.jpg"

print("Image Path:", img_path)
print("File Exists:", os.path.exists(img_path))

img = cv2.imread(img_path)

if img is None:
    print("OpenCV could not read the image.")
    exit()

print("Image loaded successfully.")

# Analyze emotion
result = DeepFace.analyze(
    img_path=img_path,
    actions=['emotion'],
    enforce_detection=False
)

# If multiple faces exist, take the first one
if isinstance(result, list):
    result = result[0]

emotion = result['dominant_emotion']

print("Detected Emotion:", emotion)

# Face coordinates
face = result['region']
x = face['x']
y = face['y']
w = face['w']
h = face['h']

# Draw rectangle
cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

# Display emotion
cv2.putText(img, emotion.upper(),
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2)

# Save output
output_path = r"C:\Users\utkar\Desktop\EmotionDetection\output.jpg"
cv2.imwrite(output_path, img)

# Show output
cv2.imshow("Emotion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Output saved at:", output_path)