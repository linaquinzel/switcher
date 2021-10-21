import cv2
import numpy as np
import os

def find_contours_of_switchers(image_path):
    """ Находим контуры. """
    # получаем и читаем картинку
    image = cv2.imread(image_path)
    image3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('contours', image3)
    cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 153, 1543], dtype="uint8")
    upper = np.array([204, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cv2.imshow("window_name", mask)
    # блюрим изображение (картинка, (размер ядра(матрицы), стандартное отклонение ядра))
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    # преобразуем картинку в чб, всем значениям >127 присваиваем 255, остальным-0
    # threshold возвращает два значения - второй переданный в функцию аргумент и картинку
    T, thresh_img = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow("ghj", thresh_img)
    # находим контуры интересных точек
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255,0,0), 3, cv2.LINE_AA, hierarchy, 1)
    cv2.imshow('contours', image)
    cv2.waitKey(0) 
    #closing all open windows 
    cv2.destroyAllWindows() 
    return contours, image

def find_coordinates_of_switchers(contours, image2):
    """ Находим координаты. """
    # словарь вида {выключатель: значение}
    switchers_coordinates = {}
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if w > 20 and h > 30:
            img_crop = image2[y - 15:y + h + 15, x - 15:x + w + 15]
            cards_name = cv2.find_features(img_crop)
            switchers_coordinates[cards_name] = (x - 15, y - 15, x + w + 15, y + h + 15)
    cv2.drawContours(image2, switchers_coordinates, -1, (255,0,0), 3, cv2.LINE_AA, 1)
    cv2.imshow("result", switchers_coordinates)
    cv2.waitKey(0)

    return switchers_coordinates

def find_features(image2):
    correct_matches_dct = {}
    directory = 'images/'
    for image in os.listdir(directory):
        img2 = cv2.imread(directory+image)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(image2, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        correct_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                correct_matches.append([m])
                correct_matches_dct[image.split('.')[0]] = len(correct_matches)
    correct_matches_dct = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    return list(correct_matches_dct.keys())[0]

def draw_rectangle_aroud_cards(switchers_coordinates, image):
    for key, value in switchers_coordinates.items():
        rec = cv2.rectangle(image, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
find_contours_of_switchers("images/box.jpg")