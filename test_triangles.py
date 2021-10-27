import cv2
import numpy as np
img = cv2.imread('test_square.jpg')  # read image from system
list1 =[]
lower = np.array([76, 0, 153], dtype="uint8")
upper = np.array([255, 204, 255], dtype="uint8")
mask = cv2.inRange(img, lower, upper)
    # блюрим изображение (картинка, (размер ядра(матрицы), стандартное отклонение ядра))
blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    # преобразуем картинку в чб, всем значениям >127 присваиваем 255, остальным-0
    # threshold возвращает два значения - второй переданный в функцию аргумент и картинку
T, returned_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('matches', img)
cv2.waitKey(0)
cv2.imshow('matches', returned_image)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(returned_image, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

def detectShape(c):  # Function to determine type of polygon on basis of number of sides
    shape = 'unknown'
    peri = cv2.arcLength(cnt, True)
    vertices = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    sides = len(vertices)
    if (sides == 3):
        shape = 'on'
    elif(sides == 4):
        x, y, w, h = cv2.boundingRect(cnt)
        aspectratio = float(w)/h
        if (aspectratio == 1):
            shape = 'square'
        else:
            shape = "off"
    elif(sides == 5):
        shape = 'pentagon'
    elif(sides == 6):
        shape = 'hexagon'
    elif(sides == 8):
        shape = 'octagon'
    elif(sides == 10):
        shape = 'star'
    else:
        shape = 'circle'
    return shape

for cnt in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[cnt])
        if 10 < w and 10 < h:
            u = (x, y)
            list1.append(u)
        for i in range(len(list1)-1):
            for j in range((len(list1)-1)-i):
                if list1[j][1] > list1[j+1][1]:
                    list1[j], list1[j+1] = list1[j+1], list1[j]
                elif list1[j][1] == list1[j+1][1]:
                    if list1[j][0] > list1[j+1][0]:
                        list1[j], list1[j+1] = list1[j+1], list1[j]
print(list1)

for cnt in contours:
    moment = cv2.moments(cnt)
    cx = int(moment['m10'] / moment['m00'])
    cy = int(moment['m01'] / moment['m00'])
    shape = detectShape(cnt)
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(img, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,0),2)  #Putting name of polygon along with the shape 
    cv2.imshow('polygons_detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()