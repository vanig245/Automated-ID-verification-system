import cv2 as cv
import numpy as np

image = cv.imread("test_id.jpg")
gray =cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3,3), 0)
edges = cv.Canny(blur, 20, 80)
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print("Number of contours:", len(contours))
# cv.drawContours(image, contours, -1, (0, 0, 0), 2)
contours = sorted(contours, key=cv.contourArea, reverse=True)
id_card_contour = None
for contour in contours:
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    area = cv.contourArea(contour)
    print("Area:", area, "Corners:", len(approx))
    if len(approx) == 4:
        id_card_contour = approx
        break
if id_card_contour is not None:
    cv.drawContours(image, [id_card_contour], -1, (0, 255, 0), 3)
    print("ID Card Detected!")
else:
    print("No ID Card Found!")
cv.imshow('smooth', image)
key = cv.waitKey(0)
print("key to press : ", key)
cv.destroyAllWindows()