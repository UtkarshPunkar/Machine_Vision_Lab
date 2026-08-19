import cv2
import pytesseract
import numpy as np

# ============================================================
# TESSERACT PATH
# ============================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ============================================================
# IMAGE PATH
# ============================================================
IMAGE_PATH = r"C:\Users\utkar\Desktop\Document vision\document.jpg"

# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================
original = cv2.imread(IMAGE_PATH)

if original is None:
    print("ERROR: Image not found!")
    exit()

# Keep original completely unchanged
image = original.copy()

print("Image loaded successfully.")


# ============================================================
# PREPROCESSING
# ============================================================

# 1. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Upscale ONLY for OCR processing
ocr_image = cv2.resize(
    gray,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC
)

# 3. Improve contrast
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(ocr_image)

# 4. Remove noise
denoised = cv2.fastNlMeansDenoising(
    enhanced,
    None,
    h=8,
    templateWindowSize=7,
    searchWindowSize=21
)

# ============================================================
# CREATE DIFFERENT OCR VERSIONS
# ============================================================

# Version 1 - Grayscale
version1 = denoised

# Version 2 - OTSU
_, version2 = cv2.threshold(
    denoised,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Version 3 - Adaptive Threshold
version3 = cv2.adaptiveThreshold(
    denoised,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    10
)


# ============================================================
# OCR FUNCTION
# ============================================================
def perform_ocr(img):

    data = pytesseract.image_to_data(
        img,
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    words = []
    confidences = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        try:
            confidence = float(data["conf"][i])
        except:
            confidence = -1

        if text and confidence > 0:
            words.append(text)
            confidences.append(confidence)

    if not confidences:
        return "", 0

    text = " ".join(words)

    confidence = sum(confidences) / len(confidences)

    return text, confidence


# ============================================================
# TEST OCR METHODS
# ============================================================

versions = {
    "Grayscale": version1,
    "OTSU": version2,
    "Adaptive Threshold": version3
}

best_text = ""
best_image = None
best_method = ""
best_confidence = -1

print("\n====================================")
print("OCR RESULTS")
print("====================================")

for name, img in versions.items():

    text, confidence = perform_ocr(img)

    print(
        f"{name:<25} "
        f"{confidence:.2f}%"
    )

    if confidence > best_confidence:

        best_confidence = confidence
        best_text = text
        best_image = img
        best_method = name


# ============================================================
# PRINT OCR RESULT
# ============================================================

print("\n====================================")
print("BEST RESULT")
print("====================================")

print("Method:", best_method)
print("Confidence:", round(best_confidence, 2), "%")

print("\nExtracted Text:\n")
print(best_text)


# ============================================================
# SAVE TEXT
# ============================================================

with open(
    "extracted_text.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(best_text)

print("\nText saved to extracted_text.txt")


# ============================================================
# SAVE PROCESSED IMAGE
# ============================================================

cv2.imwrite(
    "processed_document.png",
    best_image
)

print("Processed image saved to processed_document.png")


# ============================================================
# DISPLAY IMAGES
# ============================================================

# Function to fit image inside window
def resize_for_display(img, max_width=900, max_height=700):

    h, w = img.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1
    )

    new_width = int(w * scale)
    new_height = int(h * scale)

    return cv2.resize(
        img,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# Original image for display
display_original = resize_for_display(original)

# Processed image for display
display_processed = resize_for_display(best_image)


# ============================================================
# SHOW
# ============================================================

cv2.imshow(
    "Original Image",
    display_original
)

cv2.imshow(
    "Processed Image",
    display_processed
)

cv2.waitKey(0)
cv2.destroyAllWindows()