import cv2
import numpy as np
import os


DIRECTORY = "angle/"

def preview(image_path):
    # получаем и читаем картинку
    preimage = cv2.imread(image_path)
    # cv2.imshow('ma', preimage)
    # cv2.waitKey(0)
    image = cv2.cvtColor(preimage, cv2.COLOR_BGR2HSV)
    lower = np.array([113, 2, 115], dtype="uint8")
    upper = np.array([200, 140, 230], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    # блюрим изображение (картинка, (размер ядра(матрицы), стандартное отклонение ядра))
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    # преобразуем картинку в чб, всем значениям >127 присваиваем 255, остальным-0
    # threshold возвращает два значения - второй переданный в функцию аргумент и картинку
    T, returned_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow('matches', returned_image)
    # cv2.waitKey(0)
    return(returned_image)


def find_contours_of_switchers(image):
    """ Находим контуры. """
    # находим контуры интересных точек
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)
    cv2.imshow('ma', image)
    cv2.waitKey(0)
    return(contours)
list1 =[]
def find_coordinates_of_switchers(contours, image):
    """ Находим координаты. """
    # словарь вида {выключатель: значение}
    switchers_coordinates = {}
    # для элементов списка в диапазоне от 1 до количества найденных контуров
    for i in range(0, len(contours)):
        # (x,y) - координаты начала прямоугольника, обводящего контуры
        # (w, h) - ширина и высота этого прямоугольника
        x, y, w, h = cv2.boundingRect(contours[i])
        u = (x, y)
        list1.append(u)
        # если ширина и высота больше чем заданные значения, то
        if 10 < w < 200 and 10 < h < 60:
        # if w > 10 and h > 10:
            img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
            switchers_name = find_features(img_crop)
            switchers_coordinates[switchers_name] = (x - 15, y - 15, x + w + 15, y + h + 15)
    for i in range(len(list1)):
        for j in range((len(list1))-i):
            if list1[j][1] > list1[j+1][1]:
                list1[j], list1[j+1] = list1[j+1], list1[j]
            elif list1[j][1] == list1[j+1][1]:
                if list1[j][0] > list1[j+1][0]:
                    list1[j], list1[j+1] = list1[j+1], list1[j]
    print(list1)
    return switchers_coordinates
def find_features(image_one):
    correct_matches_dct = {}
    for image in os.listdir(DIRECTORY):
        img2 = preview(DIRECTORY+image)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img2, None)
        kp2, des2 = orb.detectAndCompute(image_one, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        correct_matches = bf.match(des2, des1)
        img3 = cv2.drawMatches(image_one, kp2, img2, kp1, correct_matches[:10], None, flags=2)
        # cv2.imshow('matches', img3)
        # cv2.waitKey(0)
        correct_matches_dct[image.split('.')[0]] = len(correct_matches)
    correct_matches_dict = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    print(list(correct_matches_dict.keys())[0])
    return list(correct_matches_dict.keys())[0]

def draw_rectangle_aroud_switchers(switchers_coordinates):
    image_ = cv2.imread("on.jpg")
    for key, value in switchers_coordinates.items():
        rec = cv2.rectangle(image_, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    cv2.imshow('Image', image_)
    cv2.waitKey(0)

image = preview("on.jpg")
contours = find_contours_of_switchers(image)
switchers_coordinates = find_coordinates_of_switchers(contours, image)
draw_rectangle_aroud_switchers(switchers_coordinates)