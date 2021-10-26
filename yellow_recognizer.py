import cv2
import numpy as np
import os


DIRECTORY = "correct_images/"

def preview_yellow(image_path):
    # получаем и читаем картинку
    preimage = cv2.imread(image_path)
    # cv2.imshow('ma', preimage)
    # cv2.waitKey(0)
    image = cv2.cvtColor(preimage, cv2.COLOR_BGR2HSV)
    lower = np.array([133, 21, 199], dtype="uint8")
    upper = np.array([203, 192, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    # блюрим изображение (картинка, (размер ядра(матрицы), стандартное отклонение ядра))
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    # преобразуем картинку в чб, всем значениям >127 присваиваем 255, остальным-0
    # threshold возвращает два значения - второй переданный в функцию аргумент и картинку
    T, returned_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow('matches', returned_image)
    # cv2.waitKey(0)
    return(returned_image)


def preview_white(image_path):
    # получаем и читаем картинку
    preimage = cv2.imread(image_path)
    # cv2.imshow('ma', preimage)
    # cv2.waitKey(0)
    image = cv2.cvtColor(preimage, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 150, 150], dtype="uint8")
    upper = np.array([204, 255, 255], dtype="uint8")
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

def find_features_yellow(image_one):
    correct_matches_dct = {}
    for image in os.listdir(DIRECTORY):
        img2 = preview_yellow(DIRECTORY+image)
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

    return list(correct_matches_dict.keys())[0]

def find_features_white(image_one):
    correct_matches_dct = {}
    for image in os.listdir(DIRECTORY):
        img2 = preview_white(DIRECTORY+image)
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

    return list(correct_matches_dict.keys())[0]

def find_coordinates_of_switchers_yellow(contours, image):
    """ Находим координаты. """
    # словарь вида {выключатель: значение}
    switchers_coordinates = {}
    # для элементов списка в диапазоне от 1 до количества найденных контуров
    for i in range(0, len(contours)):
        # (x,y) - координаты начала прямоугольника, обводящего контуры
        # (w, h) - ширина и высота этого прямоугольника
        x, y, w, h = cv2.boundingRect(contours[i])
        # если ширина и высота больше чем заданные значения, то
        if 10 < w < 200 and 10 < h < 60:
        # if w > 10 and h > 10:
            img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
            switchers_name = find_features_yellow(img_crop)
            switchers_coordinates[switchers_name] = (x - 15, y - 15, x + w + 15, y + h + 15)
    return switchers_coordinates

def find_coordinates_of_switchers_white(contours, image):
    """ Находим координаты. """
    # словарь вида {выключатель: значение}
    switchers_coordinates = {}
    # для элементов списка в диапазоне от 1 до количества найденных контуров
    for i in range(0, len(contours)):
        # (x,y) - координаты начала прямоугольника, обводящего контуры
        # (w, h) - ширина и высота этого прямоугольника
        x, y, w, h = cv2.boundingRect(contours[i])
        # если ширина и высота больше чем заданные значения, то
        if 10 < w < 200 and 10 < h < 60:
        # if w > 10 and h > 10:
            img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
            switchers_name = find_features_white(img_crop)
            switchers_coordinates[switchers_name] = (x - 15, y - 15, x + w + 15, y + h + 15)
    return switchers_coordinates

def draw_rectangle_aroud_switchers(switchers_coordinates_yellow, switchers_coordinates_white):
    image_ = cv2.imread("box7.jpg")
    for key, value in switchers_coordinates_yellow.items():
        rec = cv2.rectangle(image_, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    for key, value in switchers_coordinates_white.items():
        rec = cv2.rectangle(image_, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    cv2.imshow('Image', image_)
    cv2.waitKey(0)
    return(image_)



image_yellow = preview_yellow("box7.jpg")
image_white = preview_yellow("box7.jpg")
contours_yellow = find_contours_of_switchers(image_yellow)
contours_white = find_contours_of_switchers(image_white)
switchers_coordinates_yellow = find_coordinates_of_switchers_yellow(contours_yellow, image_yellow)
switchers_coordinates_white = find_coordinates_of_switchers_white(contours_white, image_white)
prepared_image = draw_rectangle_aroud_switchers(switchers_coordinates_yellow, switchers_coordinates_white)