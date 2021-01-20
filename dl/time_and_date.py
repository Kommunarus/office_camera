import cv2
import os
import glob
import datetime


def gettime(img_rgb, name_cam, datastart):
    if name_cam == '':
        return datastart
    if name_cam == 'кассы':
        h = 28
        y1 = 8
        y2 = 48
    else:
        h = 27
        y1 = 3
        y2 = 32

    direct = "./dl/clock/{}".format(name_cam)
    threshold = 0.85
    da = []
    if name_cam == 'кассы':
        da = [1248, 1280, 1344, 1376,   1440, 1472, 1536, 1568, 1632, 1664]

    if name_cam == 'рабочая зона':
        da = [868, 897, 955, 984,   1043, 1071, 1130, 1159, 1217, 1246]

    if name_cam == 'очередь':
        da = [432, 461, 519, 548,   607, 636, 694, 723, 781, 810]

    if name_cam == 'зал':
        da = [464, 494, 552, 581,   639, 667, 726, 755, 812, 842]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    a = []
    b = {}
    for i in range(10):
        a.append(img_gray[y1:y2, da[i]:da[i] + h])

    for i in range(10):
        list_im = sorted(glob.glob(os.path.join(direct, str(i), '*.jpg')))

        for ii, g in enumerate(list_im):
            template = cv2.imread(g, 0)

            for j in range(10):
                res = cv2.matchTemplate(a[j], template, cv2.TM_CCOEFF_NORMED)
                if res >= threshold:
                    b[j] = i

    if len(b) == 10:
        return datetime.datetime(2020, int(b[0] * 10 + b[1]), int(b[2] * 10 + b[3]), int(b[4] * 10 + b[5]),
                                 int(b[6] * 10 + b[7]), int(b[8] * 10 + b[9]))
    else:
        return datastart
