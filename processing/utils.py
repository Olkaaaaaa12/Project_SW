import numpy as np
import cv2
import imutils
from imutils import contours
from math import *
from skimage.metrics import structural_similarity


def find_con(blurred, prog, mode):
    found = False
    plate = []
    if mode == 1:
        thresh = cv2.threshold(blurred, prog, 255, cv2.THRESH_BINARY_INV)[1]
    elif mode == 2:
        win = int(blurred.shape[1] * 0.074)
        if win % 2 == 0:
            win += 1
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, win , 2)
    con = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con = imutils.grab_contours(con)
    con = sorted(con, key=cv2.contourArea, reverse=True)

    # find license plate
    for c in con:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            point1x = approx[0][0][0]
            point1y = approx[0][0][1]
            point2x = approx[1][0][0]
            point2y = approx[1][0][1]
            point3x = approx[2][0][0]
            point3y = approx[2][0][1]
            point4x = approx[3][0][0]
            point4y = approx[3][0][1]
            points = np.array([[point1x, point1y], [point2x, point2y], [point3x, point3y], [point4x, point4y]])
            x = points[np.argsort(points[:, 0]), :]
            leftMost = x[:2, :]
            rightMost = x[2:, :]
            leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
            (tl, bl) = leftMost
            rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
            (tr, br) = rightMost

            # longer and shorter edge
            val1 = sqrt(pow(tl[0] - tr[0], 2) + pow(tl[1] - tr[1], 2))
            val2 = sqrt(pow(tl[0] - bl[0], 2) + pow(tl[1] - bl[1], 2))
            x = abs(tl[1] - tr[1])
            if val1 >= (blurred.shape[1] / 3) and (val1/val2) >= 4 and (val1/val2) <= 6.5 and ((asin(x / val1) * 180 / 3.14) <= 45):
                plate = approx
                found = True
                break
    return plate, found, thresh


def perform_processing(image: np.ndarray, template, sign) -> str:
    print(f'image.shape: {image.shape}')

    #declare variables
    res = []
    signC = []

    #find contours on image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    plate, found, thresh = find_con(blurred, 120, 1)

    if found == False:
        plate, found, thresh = find_con(blurred, 135, 1)

    if found == False:
        plate, found, thresh = find_con(blurred, 0, 2)

    if found == False:
        return "PO12345"

    #sort points (left, right, top, bottom)
    point1x = plate[0][0][0]
    point1y = plate[0][0][1]
    point2x = plate[1][0][0]
    point2y = plate[1][0][1]
    point3x = plate[2][0][0]
    point3y = plate[2][0][1]
    point4x = plate[3][0][0]
    point4y = plate[3][0][1]
    points = np.array([[point1x, point1y], [point2x, point2y], [point3x, point3y], [point4x, point4y]])
    x = points[np.argsort(points[:, 0]), :]
    leftMost = x[:2, :]
    rightMost = x[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    #cut license plate from image
    input_pts = np.float32([tl, tr, br, bl])
    output_pts = np.float32([[0, 0], [400, 0], [400, 89], [0, 89]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    warped = cv2.warpPerspective(gray, M, (400, 89), flags=cv2.INTER_LINEAR)

    #find sign's contours
    thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 27, 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    con = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con = imutils.grab_contours(con)
    bound = []
    for c in con:
        #compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        #check the size of bounding box
        if w <= (warped.shape[1] / 6) and w >= 10 and h >= (warped.shape[0] * 0.50) and h <= warped.shape[0]:
            signC.append(c)
            bound.append([(x, y), (x+w, y+h)])
    if len(signC) == 0:
        return "PO12345"

    for ind, box in enumerate(bound):
        for b in bound:
            if box == b:
                continue
            else:
                if box[0][0] > b[0][0] and box[0][1] > b[0][1] and box[1][0] < b[1][0] and box[1][1] < b[1][1]:
                    signC.pop(ind)
                    bound.pop(ind)
    signC = contours.sort_contours(signC, method="left-to-right")[0]
    for c in signC:
        #extract the sign roi
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        (rH, rW) = roi.shape
        max = 0
        si = 0
        for ind, image in enumerate(template):
            image = cv2.resize(image, (rW, rH), interpolation=cv2.INTER_AREA)
            #check similarity to sign from template
            score = structural_similarity(roi, image, full=True)[0]

            #find the best match
            if score > max:
                max = score
                si = sign[ind]
        if max >= 0.26 and si != "I":
            res.append(si)
        elif si == "I":
            if cv2.mean(roi)[0] >= 220:
                res.append(si)
    if len(res) == 0:
        return "PO12345"
    st = res[0]
    for r in range(1, len(res)):
        st = st + res[r]
    return st
