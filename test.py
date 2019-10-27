import os
import cv2
import numpy as np

base_dir = os.path.dirname(__file__)
imgs_dir = os.path.join(base_dir, 'imgs')

def focus(img, radius):
    h, w, c = img.shape
    x = w // 2 - radius // 2
    y = h // 2 - radius // 2
    return x, y, radius

def squarify(img):
    y, x, c = img.shape
    smol = min(x, y)
    start_x = x // 2 - smol // 2
    start_y = y // 2 - smol // 2
    return img[start_y:start_y + smol, start_x: start_x + smol]

def adjust(img, alpha, beta):
    new_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_img[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
    return new_img

def correct(img, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

while True:
    img = cv2.imread(os.path.join(imgs_dir, 'img001.jpg'), cv2.IMREAD_COLOR)

    # Make image square and smaller
    img = squarify(img)
    img = cv2.resize(img, (100, 100))

    img = adjust(img, 2, -100)
    # img = correct(img, 0.5)
    
    # kernel = np.ones((3,3),np.uint8)

    # fX, fY, fR = focus(img, 400)
    # roi = img[fY:fY + fR, fX:fX + fR]

    # cv2.rectangle(img,(fX,fY),(fX + fR,fY + fR),(0,255,0),0)    
    # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower_skin = np.array([0,20,70], dtype=np.uint8)
    # upper_skin = np.array([20,255,255], dtype=np.uint8)

    # mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # mask = cv2.dilate(mask,kernel,iterations = 4)
    # mask = cv2.GaussianBlur(mask,(5,5),100) 

    cv2.imshow('test',img)
    # cv2.imshow('mask',mask)
    
    k = cv2.waitKey(10)
    if k == 27:
        break

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    # thresh1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)

    # cv2.imshow('result', thresh1)
