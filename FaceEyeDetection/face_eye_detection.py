import cv2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
image = cv2.imread(r"C:\Users\utkar\Desktop\FaceEyeDetection\person.jpg")
if image is None:
    print("Error: Image not found!")
    exit()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)
print("Faces Detected:", len(faces))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(
        image,
        "Face",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(20, 20)
    )
    print("Eyes Detected:", len(eyes))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(
            roi_color,
            (ex, ey),
            (ex + ew, ey + eh),
            (0, 255, 0),
            2
        )
output_path = r"C:\Users\utkar\Desktop\FaceEyeDetection\output.jpg"
cv2.imwrite(output_path, image)
print("Output image saved at:")
print(output_path)
cv2.imshow("Face and Eye Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()