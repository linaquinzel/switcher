import cv2
import numpy as np
import os

def find_contours_of_switchers(image_path):
    """ Находим контуры. """
    # получаем и читаем картинку
    image_ = cv2.imread(image_path)
    image = cv2.cvtColor(image_, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 153, 153], dtype="uint8")
    upper = np.array([204, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    # блюрим изображение (картинка, (размер ядра(матрицы), стандартное отклонение ядра))
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    # преобразуем картинку в чб, всем значениям >127 присваиваем 255, остальным-0
    # threshold возвращает два значения - второй переданный в функцию аргумент и картинку
    T, thresh_img = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    # находим контуры интересных точек
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)
    return contours, image

def find_coordinates_of_switchers(contours, image):
    """ Находим координаты. """
    # словарь вида {выключатель: значение}
    switchers_coordinates = {}
    # для элементов списка в диапазоне от 1 до количества найденных контуров
    for i in range(0, len(contours)):
        # (x,y) - координаты начала прямоугольника, обводящего контуры
        # (w, h) - ширина и высота этого прямоугольника
        x, y, w, h = cv2.boundingRect(contours[i])
        # если ширина и высота больше чем заданные значения, то
        if w > 50 and h > 50:
            img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
            switchers_name = find_features(img_crop)
            switchers_coordinates[switchers_name] = (x - 15, y - 15, x + w + 15, y + h + 15)
            return switchers_coordinates

def find_features(image_one):
    correct_matches_dct = {}
    directory = "images/"
    for image in os.listdir(directory):
        contours, img2 = find_contours_of_switchers(directory+image)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img2, None)
        kp2, des2 = orb.detectAndCompute(image_one, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        correct_matches = bf.match(des2, des1)
        img3 = cv2.drawMatches(image_one, kp2, img2, kp1, correct_matches[:10], None, flags=2)
        cv2.imshow('matches', img3)
        cv2.waitKey(0)
        correct_matches_dct[image.split('.')[0]] = len(correct_matches)
    correct_matches_dict = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    print(list(correct_matches_dict.keys())[0])

    return list(correct_matches_dict.keys())[0]

def draw_rectangle_aroud_cards(switchers_coordinates, image):
    for key, value in switchers_coordinates.items():
        rec = cv2.rectangle(image, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
contour, image = find_contours_of_switchers("images/box.jpg")
find_coordinates_of_switchers(contour, image)