import pyqrcode
import cv2
import numpy as np


def get_qr_code(data,scale):
    qr = pyqrcode.create(data, error='H', version=1, mode='binary')
    qr_code = np.array(qr.code, dtype=np.uint8)
    qr_code[qr_code == 0] = 255
    qr_code[qr_code == 1] = 0
    qr_code_size = qr_code.shape[0] * scale
    qr_code = cv2.resize(qr_code, (qr_code_size, qr_code_size), interpolation=cv2.INTER_AREA)
    return qr_code

interval = 660
scale = 12
padding = 24
qr_code = get_qr_code('',scale)
qr_code_size = qr_code.shape[0]
background = np.zeros((interval + padding * 2 + qr_code_size,interval + padding * 2 + qr_code_size), dtype=np.uint8)
background = background + 255

for x in range(0,2):
    for y in range(0, 2):
        data = str(x) + ',' + str(y)
        qr_code = get_qr_code(data,scale)
        qr_code_size = qr_code.shape[0]
        qr_x = x * interval + padding
        qr_y = y * interval + padding
        background[qr_x:qr_x + qr_code_size,qr_y:qr_y + qr_code_size] = qr_code
        #composite_image[x * crop_rows:((x + 1) * crop_rows), y * crop_cols:((y + 1) * crop_cols)] = images[key]
while True:
    cv2.imshow("background", background)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

