# import cv2 as cv
# import numpy as np
# import pytesseract
# import re
# import spacy

# image = cv.imread("test3.jpg")
# gray =cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(gray, (3,3), 0)
# edges = cv.Canny(blur, 20, 80)
# contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print("Number of contours:", len(contours))
# # cv.drawContours(image, contours, -1, (0, 0, 0), 2)
# contours = sorted(contours, key=cv.contourArea, reverse=True)
# id_card_contour = None
# for contour in contours:
#     peri = cv.arcLength(contour, True)
#     approx = cv.approxPolyDP(contour, 0.02 * peri, True)
#     area = cv.contourArea(contour)
#     print("Area:", area, "Corners:", len(approx))
#     if len(approx) == 4:
#         id_card_contour = approx
#         break
# if id_card_contour is not None:
#     cv.drawContours(image, [id_card_contour], -1, (0, 255, 0), 3)
#     print("ID card detected")

#     pts = id_card_contour.reshape(4, 2)
#     def order_points(pts):
#         rect = np.zeros((4, 2), dtype="float32")
#         s = pts.sum(axis=1)
#         rect[0] = pts[np.argmin(s)]
#         rect[2] = pts[np.argmax(s)]

#         diff = np.diff(pts, axis=1)
#         rect[1] = pts[np.argmin(diff)]
#         rect[3] = pts[np.argmax(diff)]
#         return rect

#     rect = order_points(pts)
#     tl, tr, br, bl = rect

#     width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
#     width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
#     max_width = max(int(width1), int(width2))

#     height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
#     height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
#     max_height = max(int(height1), int(height2))
#     # setA is messy setB is clean/flat
#     setA = np.array([tl, tr, br, bl], dtype="float32")
#     setB = np.array([[0,0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype="float32")

#     matrix = cv.getPerspectiveTransform(setA, setB)
#     warped_image = cv.warpPerspective(image, matrix, (max_width, max_height))

#     warped_gray = cv.cvtColor(warped_image, cv.COLOR_BGR2GRAY)
#     warped_blur = cv.GaussianBlur(warped_gray, (3,3), 0)

#     binary_image = cv.adaptiveThreshold(
#         src= warped_blur,
#         maxValue=255,
#         adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv.THRESH_BINARY,
#         blockSize=11,
#         C=5
#     )

#     ocr_text = pytesseract.image_to_string(binary_image, config='--psm 6')
#     # print("extracted text")
#     # print(ocr_text)

#     id_count = r"\d{9}"
#     id_pattern = re.search(id_count, ocr_text)

#     id_date_count = r"\d{2}/\d{2}/\d{4}"
#     id_date_pattern = re.findall(id_date_count, ocr_text)
#     # print(id_date_pattern, id_pattern)
#     nlp = spacy.load("en_core_web_sm")
#     docs = nlp(ocr_text)
#     extracted_name = "Not Found"
#     for ent in docs.ents:
#         if ent.label_ == "PERSON":
#             if len(ent.text) > 2 and "DRIVER" not in ent.text:
#                 extracted_name = ent.text
#                 break
#     if extracted_name == "Not Found":
#         uppercase_words = re.findall(r"[A-Z]{3,}", ocr_text)
#         ignore_list = ["DRIVER", "LICENSE", "DOB", "EXP", "SEX", "CLASS", "ANYTOWN", "ANYSTREET"]
#         possible_names = [word for word in uppercase_words if word not in ignore_list]
        
#         if possible_names:
#             extracted_name = " ".join(possible_names[:2])

#     kyc_profile = {
#         "Document Type": "Driver License" if "DRIVER" in ocr_text.upper() else "Unknown",
#         "Extracted Name": extracted_name,
#         "ID Number": id_pattern.group(0) if id_pattern else "Not Found",
#         "Dates Found": id_date_pattern if id_date_pattern else "Not Found"
#     }
#     print("FINAL STRUCTURED KYC PROFILE")
#     for key, value in kyc_profile.items():
#         print(f"{key}: {value}")
#     cv.imshow('smooth', binary_image)
#     key = cv.waitKey(0)
#     print("key to press : ", key)
#     cv.destroyAllWindows()

# else:
#     print("no ID card found")

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
    """Upscale + adaptive threshold tuned to clear background patterns."""
    scale = 2.0
    upscaled = cv.resize(gray_image, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    blur = cv.GaussianBlur(upscaled, (3, 3), 0)
    
    # Increased block size and constant C to clean complex backgrounds/guilloche noise
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

        # Adjusted sanity check to prevent cropping internal blocks on pre-cropped images
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

    # Switched to PSM 11 (Sparse text localization) to scan dispersed fields uniformly
    ocr_text = pytesseract.image_to_string(binary_image, config="--psm 11")

    # --- Extraction Logic Adjusted for Aadhaar Format ---
    # Matches 12-digit sequences formatted with spaces (XXXX XXXX XXXX) or continuous strings
    id_pattern = re.search(r"\d{4}\s\d{4}\s\d{4}|\d{12}", ocr_text)
    id_date_pattern = re.findall(r"\d{2}/\d{2}/\d{4}", ocr_text)

    extracted_name = "Not Found"
    doc = nlp(ocr_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if len(ent.text) > 2 and not any(k in ent.text.upper() for k in ["GOVERNMENT", "INDIA", "MALE", "FEMALE"]):
                extracted_name = ent.text
                break

    if extracted_name == "Not Found":
        ignore_list = [
            "GOVERNMENT", "INDIA", "UNIQUE", "IDENTIFICATION", "AUTHORITY",
            "MALE", "FEMALE", "DOB", "YEAR", "ENROLLMENT", "HUSBAND", "WIFE", "FATHER"
        ]
        uppercase_words = re.findall(r"[A-Z]{3,}", ocr_text)
        possible_names = [w for w in uppercase_words if w not in ignore_list]
        if possible_names:
            extracted_name = " ".join(possible_names[:2])

    # Classification Layer
    upper_text = ocr_text.upper()
    if "UNIQUE IDENTIFICATION" in upper_text or "GOVERNMENT OF INDIA" in upper_text or "AADHAAR" in upper_text or "AADHAR" in upper_text:
        doc_type = "Aadhaar Card"
    elif "DRIVER" in upper_text or "LICENSE" in upper_text:
        doc_type = "Driver License"
    elif "PASSPORT" in upper_text:
        doc_type = "Passport"
    elif "INCOME TAX" in upper_text or "PAN" in upper_text:
        doc_type = "PAN Card"
    else:
        doc_type = "Unknown"

    # Generic Address Matching Fallback
    address_match = re.search(
        r"(?:C/O|W/O|D/O|S/O)?[^\n]+",
        ocr_text,
        re.IGNORECASE,
    )

    return {
        "document_type": doc_type,
        "extracted_name": extracted_name,
        "id_number": id_pattern.group(0) if id_pattern else "Not Found",
        "dates_found": id_date_pattern if id_date_pattern else [],
        "address": address_match.group(0).strip() if address_match else "Not Found",
        "raw_text": ocr_text,
        "card_detected": card_detected,
    }