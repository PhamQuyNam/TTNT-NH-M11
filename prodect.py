
import numpy as np

import tensorflow as tf
import cv2
from datetime import datetime

# print(tf.__version__)
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

saved_model = tf.keras.models.load_model("BienSo_28.h5")

timeStart = datetime.now()
print(datetime.now())
bienanh = cv2.imread('./kytucut/kitu 0.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('0 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 1.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('1 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 2.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('2 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 3.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('3 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 4.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('4 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 5.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('5 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 6.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('6 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 7.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('7 nhãn = ', final)
# plt.imshow(bienanh)

bienanh = cv2.imread('./kytucut/kitu 8.jpg')
bienanh = cv2.resize(bienanh, (28, 28))
bienanh = np.array(bienanh)
result = saved_model.predict(bienanh.reshape(1, 28, 28, 3))
final = np.argmax(result)
# print('classNames = ' + str(final))
final = classNames[final]
print('8 nhãn = ', final)
# plt.imshow(bienanh)

timeStop = datetime.now()
print(datetime.now())

print('timeRun = ', timeStop - timeStart)
