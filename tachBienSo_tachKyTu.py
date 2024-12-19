import cv2
import numpy as np
import os
import tensorflow as tf

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
dem = 0
for c in contours:
    # peri = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.03 * peri, True)
    # area = cv2.contourArea(c)
    # x, y, w, h = cv2.boundingRect(c)
    # aspectRatio = w / float(h)
    # if len(approx) == 4 and 1.1 <= aspectRatio <= 1.5:
    #     screenCnt = approx
    #     anh_cut = img[y:(y + h), x:(x + w)]
    #     anh_kytu_gray = cv2.cvtColor(anh_cut, cv2.COLOR_RGB2GRAY)
    #     anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]
    #     anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(760, 560))
    #     cv2.imwrite('Biensocut/' + "motorcycle_1" + str(dem) + ".jpg", anh_kytu_bw)
    #     break

    x, y, w, h = cv2.boundingRect(c)
    aspectRatio = w / float(h)
    area = cv2.contourArea(c)
    if 1.1 <= aspectRatio <= 1.5 and area > 1000:
        screenCnt = c
        anh_cut = img[y:(y + h), x:(x + w)]
        anh_kytu_gray = cv2.cvtColor(anh_cut, cv2.COLOR_RGB2GRAY)
        anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]
        anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(760, 560))
        cv2.imwrite('Biensocut/' + "motorcycle" + str(dem) + ".jpg", anh_kytu_bw)
        break
dem = dem + 1

# Tách ký tự
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
path_folder = "Biensocut/"

dem = 0
images = []
labels = []
toado = []
ketqua = []
file_list = os.listdir(path_folder)
for img_item in file_list:
    img = cv2.imread(os.path.join(path_folder, img_item))
    labels.append(str(img_item))
    images.append(img)
    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh1 = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # dem = 1
    label = str(img_item)
    label = label[:(len(label) - 4)]
    saved_model = tf.keras.models.load_model("BienSo_28.h5")

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        solidity = cv2.contourArea(c) / float(w * h)
        heightRatio = h / float(img.shape[0])
        if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.3 < heightRatio < 2.0:
            anh_kytu = img[y:(y + h), x:(x + w)]
            anh_kytu_gray = cv2.cvtColor(anh_kytu, cv2.COLOR_RGB2GRAY)
            anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]

            anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(28, 28))
            anh_kytu_bw = np.array(anh_kytu_bw)
            anh_kytu_bw = np.repeat(anh_kytu_bw[:, :, np.newaxis], 3, axis=-1)  # Convert to 3-channel image
            anh_kytu_bw = anh_kytu_bw[np.newaxis, :, :, :]  # Add a new axis for batch size

            result = saved_model.predict(anh_kytu_bw)
            final = np.argmax(result)
            ketqua.append(result)

            anh_kytu_gray = cv2.cvtColor(anh_kytu, cv2.COLOR_BGR2GRAY)
            anh_kytu_bw = cv2.threshold(anh_kytu_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            anh_kytu_bw = cv2.bitwise_not(anh_kytu_bw)  # Invert the image

            if anh_kytu_bw is not None and anh_kytu_bw.shape[0] > 0 and anh_kytu_bw.shape[1] > 0:
                anh_kytu_bw = cv2.resize(anh_kytu_bw, (112, 112))  # Resize to a fixed size
                cv2.imwrite('kytucut/' + "kitu " + str(dem) + ".jpg", anh_kytu_bw)

                # Display the character image
                cv2.imshow("Character", anh_kytu_bw)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                dem = dem + 1
            else:
                print("Error: anh_kytu_bw is invalid")

#nhan diện ký tự

print("Done")


