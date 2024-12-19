
import numpy as np

import tensorflow as tf
import cv2
import os
from datetime import datetime


class License_plate_code:
    classNames = {0: '0',
                  1: '1',
                  2: '2',
                  3: '3',
                  4: '4',
                  5: '5',
                  6: '6',
                  7: '7',
                  8: '8',
                  9: '9',
                  10: 'A',
                  11: 'B',
                  12: 'C',
                  13: 'D',
                  14: 'E',
                  15: 'F',
                  16: 'G',
                  17: 'H',
                  18: 'K',
                  19: 'L',
                  20: 'M',
                  21: 'N',
                  22: 'P',
                  23: 'R',
                  24: 'S',
                  25: 'T',
                  26: 'U',
                  27: 'V',
                  28: 'X',
                  29: 'Y',
                  30: 'Z',
                  }

# phát hiện biển số xe
def detectPlate():
    img = cv2.imread("./RealData/2.jpg")

    # img = cv2.resize(img, dsize=(472, 303))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)
    img_gamma = cv2.convertScaleAbs(img_gray, alpha=0.5, beta=0)  # Gamma correction
    img_eq = cv2.equalizeHist(img_gamma)  # Histogram equalization

    # Thresholding
    thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
    equal_histogram = cv2.equalizeHist(noise_removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=20)
    sub_morp_image = cv2.subtract(equal_histogram, morph_image)
    ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)
    canny_image = cv2.Canny(thresh_image, 250, 255)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        area = cv2.contourArea(c)
        if 1.1 <= aspectRatio <= 1.5 and area > 1000:
            screenCnt = c

            break

    if screenCnt is None:
        print("Can't detect!")
        return
    anh_kytu = img[y:(y + h), x:(x + w)]
    anh_kytu_gray = cv2.cvtColor(anh_kytu, cv2.COLOR_RGB2GRAY)
    anh_kytu_bw1 = cv2.adaptiveThreshold(anh_kytu_gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    anh_kytu_bw = cv2.resize(anh_kytu_bw1, dsize=(760, 560))
    cv2.imwrite('reservedData/' + "12435.jpg", anh_kytu_bw)
    cv2.imwrite('reservedData/' + "1243.jpg", anh_kytu)
    # cv2.imshow("bienso1", eroded)

    cv2.imshow("bienso", anh_kytu_bw)
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("original_image", img)
    cv2.imshow("processed_image", anh_kytu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    segmentChar(anh_kytu_bw)


def segmentChar(img):
    position = []
    list_img = []
    dem = 0
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        solidity = cv2.contourArea(c) / float(w * h)
        heightRatio = h / float(img.shape[0])
        if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.3 < heightRatio < 2.0:
            anh_kytu = img[y:(y + h), x:(x + w)]
            anh_kytu = cv2.resize(anh_kytu, dsize=(28, 28))
            cv2.imwrite('kytucut/' + "kitu_ " + str(dem) + ".jpg", anh_kytu)
            list_img.append(anh_kytu)
            position.append([x, y, dem])
            dem += 1
            print('segmentchar')
    sort_position(position)
    predictChar(list_img)


def predictChar(list_img):
    global strPlate
    global first_line
    global second_line
    global saved_model

    list_str_char = []
    class_plate = License_plate_code()
    for imgChar in list_img:
        img_cvt_rgb = cv2.cvtColor(imgChar, cv2.COLOR_GRAY2RGB)
        img_cvt_rgb = np.array(img_cvt_rgb)

        result = saved_model.predict(img_cvt_rgb.reshape(1, 28, 28, 3))
        final = np.argmax(result)
        final = class_plate.classNames[final]
        list_str_char.append(final)
    list_str_char = np.array(list_str_char)

    list_char = np.concatenate([list_str_char[first_line[:, 2]], list_str_char[second_line[:, 2]]])
    strPlate = list_char[0] + list_char[1] + '-' + list_char[2] + list_char[3] + ' '
    if np.count_nonzero(list_char) == 9:
        strPlate += list_char[4] + list_char[5] + list_char[6] + '.' + list_char[7] + list_char[8]
    elif np.count_nonzero(list_char) == 8:
        strPlate += list_char[4] + list_char[5] + list_char[6] + list_char[7]
    else:
        strPlate = "Can't detect"
    print('Bien so xe: ' + strPlate)


def sort_position(position):
    global first_line
    global second_line
    position = np.array(position)
    position = position[np.argsort(position[:, 1])]
    # print(position)

    first_line = position[0:4]
    second_line = position[4:]
    first_line = first_line[np.argsort(first_line[:, 0])]
    second_line = second_line[np.argsort(second_line[:, 0])]


if __name__ == '__main__':

    # Tách ký tự
    saved_model = tf.keras.models.load_model("BienSo_28.h5")
    # position = []
    first_line = []
    second_line = []
    timeStart = datetime.now()
    strPlate = ''
    detectPlate()
    timeStop = datetime.now()
    print('timeRun = ', timeStop - timeStart)

