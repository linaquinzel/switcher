import cv2
import numpy as np

list2 = []
list3 = []

lower = np.array([30, 1, 1], dtype="uint8")
upper = np.array([100, 50, 60], dtype="uint8")
lower_green = np.array([10, 100, 10], dtype="uint8")
upper_green = np.array([30, 254, 30], dtype="uint8")

def detectShape(cnt):  # Function to determine type of polygon on basis of number of sides
    shape = 'unknown'
    peri = cv2.arcLength(cnt, True)
    vertices = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    sides = len(vertices)
    if (sides == 3):
        shape = 'on'
        list2.append(1)
    elif(sides == 4):
        shape = 'off'
        list2.append(0)
    return shape

def sorter(lists):
    for i in range(len(lists)-1):
        for j in range((len(lists)-1)-i):
            if lists[j][1] > lists[j+1][1]:
                lists[j], lists[j+1] = lists[j+1], lists[j]
            elif lists[j][1] == lists[j+1][1]:
                if lists[j][0] > lists[j+1][0]:
                    lists[j], lists[j+1] = lists[j+1], lists[j]
    for kl in lists:
        list3.append(kl[2])
    return(list3)


video_capture = cv2.VideoCapture(0)
def switcher_recognizer(lower, upper):
    list1 = []
    ret, frame = video_capture.read()
    if not ret:
        pass
    img = frame[:, :, ::-1]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(rgb, lower, upper)
    # блюрим изображение (картинка, (размер ядра(матрицы), стандартное отклонение ядра))
    blurred = cv2.GaussianBlur(mask, (1, 1), 0)
    # преобразуем картинку в чб, всем значениям >200 присваиваем 255, остальным-0
    # threshold возвращает два значения - второй переданный в функцию аргумент и картинку
    T, returned_image = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('matches', img)
    cv2.waitKey(0)
    cv2.imshow('ret', returned_image)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(returned_image, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

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
    for cn in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[cn])
        if 1 < w and 1 < h:
            u = (x, y, list2[cn])
            list1.append(u)
    return(list1)
condition = False
switchers_list = switcher_recognizer(lower, upper)
switchers_list.append(switcher_recognizer(lower_green, upper_green))
if len(switchers_list) != 8:
    condition = True
else:
    print(list3)
while condition == True:
    switchers_list = switcher_recognizer()
    if len(switchers_list) != 8:
        condition = True
    else:
        condition = False
        list4 = sorter(list3)
        print(list4)
