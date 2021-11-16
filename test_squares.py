import sys
import numpy as np
import cv2 as cv


hsv_min = np.array([160, 160, 40], dtype="uint8")
hsv_max = np.array([180, 180, 60], dtype="uint8")
list1 = []
# hsv_min = np.array((194, 170, 251), np.uint8)
# hsv_max = np.array((200, 172, 255), np.uint8)

if __name__ == '__main__':
    fn = 'triangles_and_squares.jpg' # имя файла, который будем анализировать
    img = cv.imread(fn)

    hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV ) # меняем цветовую модель с BGR на HSV
    thresh = cv.inRange( hsv, hsv_min, hsv_max ) # применяем цветовой фильтр
    contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    counter = 0

    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv.minAreaRect(cnt) # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect) # поиск четырех вершин прямоугольника
        box = np.int0(box) # округление координат
        area = int(rect[1][0]*rect[1][1]) # вычисление площади
        if 1000 < area:
            cv.drawContours(img,[box],0,(255,0,0),2) # рисуем прямоугольник
    for cnt in range(0, len(contours0)):
        x, y, w, h = cv.boundingRect(contours0[cnt])
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
    cv.imshow('contours', img) # вывод обработанного кадра в окно

    cv.waitKey()
    cv.destroyAllWindows()