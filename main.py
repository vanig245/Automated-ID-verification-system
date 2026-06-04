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

pts = id_card_contour.reshape(4, 2)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

rect = order_points(pts)
tl, tr, br, bl = rect

width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
max_width = max(int(width1), int(width2))

height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
max_height = max(int(height1), int(height2))
    # setA is messy setB is clean/flat
setA = np.array([tl, tr, br, bl], dtype="float32")
setB = np.array([[0,0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype="float32")

matrix = cv.getPerspectiveTransform(setA, setB)
warped_image = cv.warpPerspective(image, matrix, (max_width, max_height))
cv.imshow('smooth', warped_image)
key = cv.waitKey(0)
print("key to press : ", key)
cv.destroyAllWindows()