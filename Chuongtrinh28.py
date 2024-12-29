import tkinter.messagebox

import cv2
from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
# import tensorflow as tf
import pickle
import numpy as np
# import matplotlib.pyplot as plt
from datetime import *
import datetime as dtime
# from tensorflow.keras.models import Model
from tensorflow.keras import models
import pandas as pd
# import time


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


def detectPlate(img):
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

    thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)[1]
    canny_image = cv2.Canny(thresh_image, 250, 255)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)

    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in contours:
        # peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        # x, y, w, h = cv2.boundingRect(c)
        # aspectRatio = w / float(h)
        # if len(approx) == 4 and 1.1 <= aspectRatio <= 1.5:
        #     screenCnt = approx
        #     break
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
    # anh_kytu_bw = cv2.threshold(anh_kytu_gray, 150, 255, cv2.THRESH_BINARY)[1]
    anh_kytu_bw = cv2.adaptiveThreshold(anh_kytu_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    anh_kytu_bw = cv2.resize(anh_kytu_bw, dsize=(760, 560))
    cv2.imwrite('reservedData/' + "01235.jpg", anh_kytu_bw)
    # cv2.imshow("bienso", anh_kytu)
    segmentChar(anh_kytu_bw)


def segmentChar(img):
    print('Dang tach ky tu')
    position = []
    list_img = []
    count_char = 0
    dem = 0
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        solidity = cv2.contourArea(c) / float(w * h)
        heightRatio = h / float(img.shape[0])
        # if 0.1 < aspectRatio < 1.0 and solidity > 0.1 and 0.3 < heightRatio < 2.0:
        if 0.2 < aspectRatio < 0.8 and solidity > 0.5 and 0.3 < heightRatio < 1.0:
            anh_kytu = img[y:(y + h), x:(x + w)]
            anh_kytu = cv2.resize(anh_kytu, dsize=(28, 28))
            cv2.imwrite('kytucut/' + "kitu_ " + str(dem) + ".jpg", anh_kytu)
            list_img.append(anh_kytu)
            position.append([x, y, count_char])
            count_char += 1
            dem +=1
    if 8 <= count_char <= 9:
        sort_position(position)
        predictChar(list_img, count_char)
    else:
        root.textPlate.config(text="Can't detect")
        print("Can't detect! (Nhan dien ky tu)")


def predictChar(list_img, count_char):
    global strPlate
    global first_line
    global second_line
    global saved_model
    print('Dang predict ky tu')
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
    if count_char == 9:
        strPlate += list_char[4] + list_char[5] + list_char[6] + '.' + list_char[7] + list_char[8]
    elif count_char == 8:
        strPlate += list_char[4] + list_char[5] + list_char[6] + list_char[7]
    else:
        strPlate = "Can't detect"
    root.textPlate.config(text=strPlate)
    print('Done')


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


# tạo các widgets
def createwidgets():

    root.feedlabel = Label(root, bg="steelblue", fg="white", text="CAMERA", font=('sans-serif', 20))
    root.feedlabel.grid(row=0, column=0, padx=10, pady=10)
    # root.feedlabel.place(x=5, y=5)

    root.previewlabel = Label(root, bg="steelblue", fg="white", text="PICTURE", font=('sans-serif', 20))
    root.previewlabel.grid(row=0, column=5, padx=10, pady=10)

    root.cameraLabel = Label(root, bg="steelblue", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=1, column=0, padx=10, pady=10)

    root.imageLabel = Label(root, bg="steelblue", borderwidth=3, relief="groove")
    root.imageLabel.grid(row=1, column=5, padx=10, pady=10, columnspan=1)
    saved_image = Image.open('./reservedData/noImage.jpg')
    saved_image = ImageTk.PhotoImage(saved_image)
    root.imageLabel.config(image = saved_image)
    root.imageLabel.photo = saved_image

    root.captureBTN = Button(root, text="CHỤP ẢNH", command=Capture, bg="LIGHTBLUE", font=('sans-serif', 15), width=20)
    root.captureBTN.grid(row=2, column=0, padx=10, pady=10)

    root.textPlate = Label(root, bg="steelblue", fg="white", text="Can't detect", font=('sans-serif', 20), width=20)
    root.textPlate.grid(row=2, column=5, padx=10, pady=10)

    root.btnXeVao = Button(root, text="XE VÀO", command=MotoInput, bg="green", font=('sans-serif', 15), width=20)
    root.btnXeVao.grid(row=3, column=0, padx=10, pady=10)

    root.textMoney = Label(root, bg="steelblue", fg="white", text="0", font=('sans-serif', 20), width=20)
    root.textMoney.grid(row=3, column=5, padx=10, pady=10)

    root.btnXeRa = Button(root, text="XE RA", command=MotoOutput, bg="red", font=('sans-serif', 15), width=20)
    root.btnXeRa.grid(row=4, column=0, padx=10, pady=10)

    # Gọi hàm ShowFeed
    ShowFeed()


def MotoInput():
    timeStart = datetime.now()
    global checkPlate
    checkPlate = True

    # Kiểm tra điều kiện nếu biển số không phát hiện được
    if strPlate == "Can't detect":
        tk.messagebox.showerror('Cảnh báo', 'Chưa có xe nào được ghi nhận!')
    else:
        # Đọc dữ liệu từ file CSV
        data_XeVao = pd.read_csv("./reservedData/CacheFile.csv")

        # Tạo một DataFrame mới với dữ liệu cần thêm
        new_data = pd.DataFrame([{
            'key': strPlate,
            'timein': dateImage,
            'timeout': 'None',
            'pathin': pathImage,
            'pathout': 'None',
            'money': '0'
        }])

        # Nối dữ liệu mới vào DataFrame hiện tại
        data_XeVao = pd.concat([data_XeVao, new_data], ignore_index=True)

        # Ghi dữ liệu đã cập nhật vào file CSV
        data_XeVao.to_csv('./reservedData/CacheFile.csv', index=False)

        # Cập nhật giao diện người dùng
        root.textMoney.config(text="0")

    timeStop = datetime.now()
    print('timeRun = ', timeStop - timeStart)


def MotoOutput():
    global checkPlate
    checkPlate = True
    timeStart = datetime.now()

    if strPlate == "Can't detect":
        tk.messagebox.showerror('Cảnh báo', 'Chưa có xe nào được ghi nhận!')
    else:
        # Đọc dữ liệu từ các file CSV
        cache = pd.read_csv("./reservedData/CacheFile.csv")
        archive = pd.read_csv("./reservedData/ArchiveFile.csv")

        # Tìm và cập nhật thông tin xe trong cache
        for i in range(cache.shape[0]):
            if cache.at[i, 'key'] == strPlate:
                cache.at[i, 'timeout'] = dateImage
                cache.at[i, 'pathout'] = pathImage
                pay_time = pay_bill(cache.at[i, 'timein'], dateImage)
                cache.at[i, 'money'] = pay_time

                # Thêm bản ghi vào archive
                archive = pd.concat([archive, cache.loc[cache['key'] == strPlate]], ignore_index=True)

                # Xóa bản ghi trong cache
                cache = cache[cache.key != strPlate]

                # Cập nhật giao diện người dùng
                root.textMoney.config(text=str(pay_time))
                break

        # Ghi dữ liệu đã cập nhật vào các file CSV
        cache.to_csv('./reservedData/CacheFile.csv', index=False)
        archive.to_csv('./reservedData/ArchiveFile.csv', index=False)

    timeStop = datetime.now()
    print('timeRun = ', timeStop - timeStart)


def pay_bill(timein, timeout):
    pay_time = datetime.strptime(timeout, "%d/%m/%Y %H:%M:%S") - datetime.strptime(timein, "%d/%m/%Y %H:%M:%S")
    if pay_time < dtime.timedelta(seconds=21600):
        return 5000
    elif pay_time < dtime.timedelta(seconds=43200):
        return 10000
    elif pay_time < dtime.timedelta(seconds=86400):
        return 15000
    else:
        return 15000 + (pay_time.days * 10000)


# Hiển thị Camera dưới Label
def ShowFeed():
    # Chụp từng khung hình
    ret, frame = root.cap.read()

    if ret:
        # để khung hình theo chiều dọc

        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                    (0, 255, 255))
        # chuyển thành màu RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Tạo bộ nhớ hình ảnh từ giao diện mảng xuất khung ở trên
        videoImg = Image.fromarray(cv2image)

        # Tạo đối tượng của lớp PhotoImage () để hiển thị khung
        imgtk = ImageTk.PhotoImage(image=videoImg)

        # cấu hình label để hiển thị khung cam
        root.cameraLabel.configure(image=imgtk)

        # Keeping a reference
        root.cameraLabel.imgtk = imgtk

        # gọi hàm sau 1ms
        root.cameraLabel.after(1, ShowFeed)
    else:
        # cấu hình label để hiển thị khung cam
        root.cameraLabel.configure(image='')


def Capture():
    global dateImage
    global pathImage
    global checkPlate

    timeStart = datetime.now()
    if checkPlate:
        # Lưu trữ ngày ở định dạng đã đề cập trong biến image_name
        dateImage = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        pathImage = datetime.now().strftime('%d-%m-%Y %H-%M-%S') + '.jpg'
        checkPlate = False

    # Chụp khung hình
    frame = root.cap.read()[1]

    # Hiển thị ngày giờ lên khung
    cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                (0, 255, 255))

    # Ghi ảnh với khung đã chụp. Hàm trả về Giá trị Boolean được lưu trữ trong biến thành công
    cv2.imwrite('./dataCamera/' + pathImage, frame)

    # Mở ảnh đã lưu bằng cách sử dụng open () của lớp Image lấy ảnh đã lưu làm đối số
    saved_image = Image.open('./dataCamera/' + pathImage)
    # saved_image = Image.open('./reservedData/381.jpg')

    # Tạo đối tượng của lớp PhotoImage () để hiển thị khung
    saved_image = ImageTk.PhotoImage(saved_image)

    # cấu hình label để hiển thị khung cam
    root.imageLabel.config(image=saved_image)

    # Keeping a reference
    root.imageLabel.photo = saved_image

    # Nhan dien
    detectPlate(frame)
    # detectPlate(cv2.imread('./reservedData/381.jpg'))

    timeStop = datetime.now()
    print('timeRun = ', timeStop - timeStart)


if __name__ == "__main__":
    # Kiểm tra ảnh đầu tiên của xe
    checkPlate = True
    # Đường dẫn ảnh
    dateImage = ''
    pathImage = './dataCamera'

    saved_model = models.load_model("BienSo_28.h5")
    first_line = []
    second_line = []
    # timeStart = datetime.now()
    strPlate = "Can't detect"

    # timeStop = datetime.now()
    # print('timeRun = ', timeStop - timeStart)

    # Tao doi tuong cho lop Tk
    root = tk.Tk()
    # Khoi tao doi  tuong Camera
    root.cap = cv2.VideoCapture(0)

    # Set chieu cao va chieu rong
    # width, height = 1280, 720
    # width, height = 1920, 1080
    width, height = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # set ten form, window size, mau me
    root.title("Nhận diện biển số xe")
    root.geometry("1366x768")
    root.resizable(True, True)
    root.configure(background="sky blue")

    # Gọi hàm khởi tạo giao diện và chạy chương trình
    createwidgets()

    root.mainloop()
