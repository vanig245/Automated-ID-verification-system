import cv2 as cv
import numpy as np
import pytesseract
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def preprocess_for_ocr(gray_image):
    scale = 2.0
    upscaled = cv.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    blur = cv.GaussianBlur(upscaled, (3, 3), 0)
    
    binary = cv.adaptiveThreshold(
        src=blur,
        maxValue=255,
        adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv.THRESH_BINARY,
        blockSize=51,  
        C=12,          
    )
    return binary


def process_id_image(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)

    if image is None:
        return {"error": "Could not decode image. Please upload a valid JPG/PNG."}

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur, 20, 80)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    id_card_contour = None
    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        area = cv.contourArea(contour)
        if len(approx) == 4 and area > (img_area * 0.10):
            id_card_contour = approx
            break

    card_detected = id_card_contour is not None

    if card_detected:
        pts = id_card_contour.reshape(4, 2)
        rect = order_points(pts)
        tl, tr, br, bl = rect

        width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width1), int(width2))

        height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height1), int(height2))

        warp_area = max_width * max_height
        if warp_area < (img_area * 0.75):
            card_detected = False

    if card_detected:
        setA = np.array([tl, tr, br, bl], dtype="float32")
        setB = np.array(
            [[0, 0], [max_width - 1, 0],
             [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype="float32",
        )
        matrix = cv.getPerspectiveTransform(setA, setB)
        warped = cv.warpPerspective(image, matrix, (max_width, max_height))
        warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    else:
        warped_gray = gray

    binary_image = preprocess_for_ocr(warped_gray)
    ocr_text = pytesseract.image_to_string(binary_image, config="--psm 11")

    id_pattern = re.search(r"\d{4}\s\d{4}\s\d{4}|\d{12}", ocr_text)
    id_date_pattern = re.findall(r"(?:0[1-9]|[12][0-9]|3[01])(?:/|-)(?:0[1-9]|1[0-2])(?:/|-)(?:19|20)\d{2}", ocr_text)

    extracted_name = "Not Found"
    doc = nlp(ocr_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if len(ent.text) > 2 and not any(k in ent.text.upper() for k in ["GOVERNMENT", "INDIA", "MALE", "FEMALE"]):
                extracted_name = ent.text
                break

    if extracted_name == "Not Found":
        ignore_list = [
            "GOVERNMENT", "INDIA", "UNIQUE", "IDENTIFICATION", "AUTHORITY", "AADHAAR",
            "MALE", "FEMALE", "DOB", "YEAR", "ENROLLMENT", "HUSBAND", "WIFE", "FATHER"
        ]
        uppercase_words = re.findall(r"[A-Z]{3,}", ocr_text)
        possible_names = [w for w in uppercase_words if w not in ignore_list]
        if possible_names:
            extracted_name = " ".join(possible_names[:2])

    upper_text = ocr_text.upper()
    if "UNIQUE IDENTIFICATION" in upper_text or "GOVERNMENT OF INDIA" in upper_text or "AADHAAR" in upper_text or "AADHAR" in upper_text:
        doc_type = "Aadhaar Card"
    # elif "DRIVER" in upper_text or "LICENSE" in upper_text:
    #     doc_type = "Driver License"
    # elif "PASSPORT" in upper_text:
    #     doc_type = "Passport"
    # elif "INCOME TAX" in upper_text or "PAN" in upper_text:
    #     doc_type = "PAN Card"
    else:
        doc_type = "Unknown"

    return {
        "document_type": doc_type,
        "extracted_name": extracted_name,
        "id_number": id_pattern.group(0) if id_pattern else "Not Found",
        "dates_found": id_date_pattern if id_date_pattern else [],
        "raw_text": ocr_text,
        "card_detected": card_detected,
    }