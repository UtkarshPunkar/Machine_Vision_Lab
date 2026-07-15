import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r"C:\Users\utkar\Desktop\Metal_Defect_Detection\metal.jpg"

img = cv2.imread(image_path)

if img is None:
    print("ERROR: Image not found!")
    print("Check the image path.")
    exit()

cv2.imshow("1. Original Image", img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("2. Grayscale Image", gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("3. Blurred Image", blur)

edges = cv2.Canny(blur, 50, 150)

cv2.imshow("4. Edge Detection", edges)

_, thresh = cv2.threshold(
    blur,
    120,
    255,
    cv2.THRESH_BINARY_INV
)

cv2.imshow("5. Threshold Image", thresh)

kernel = np.ones((3,3), np.uint8)

opening = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

cv2.imshow("6. Opening", opening)

closing = cv2.morphologyEx(
    opening,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

cv2.imshow("7. Closing", closing)

segmented = closing.copy()

cv2.imshow("8. Segmented Defects", segmented)

contours, _ = cv2.findContours(
    segmented,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour_img = img.copy()

cv2.drawContours(
    contour_img,
    contours,
    -1,
    (0, 0, 255),
    2
)

cv2.imshow("9. All Contours", contour_img)

result = img.copy()

defect_count = 0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > 20:

        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            "Defect",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )
        defect_count += 1

cv2.imshow("10. Final Defect Detection", result)

print("=" * 40)
print("Metal Surface Defect Detection")
print("=" * 40)
print("Total Defects Detected :", defect_count)
print("=" * 40)

output_path = r"C:\Users\utkar\Desktop\Metal_Defect_Detection\Detected_Defects.jpg"

cv2.imwrite(output_path, result)

print("Result saved at:")
print(output_path)
titles = [
    "Original",
    "Gray",
    "Blur",
    "Edges",
    "Threshold",
    "Opening",
    "Closing",
    "Final Detection"
]

images = [
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    gray,
    blur,
    edges,
    thresh,
    opening,
    closing,
    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(16,10))

for i in range(len(images)):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()