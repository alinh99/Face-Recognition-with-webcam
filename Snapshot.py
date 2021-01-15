import cv2
import numpy as np
import os
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
                help="path to train image")

args = vars(ap.parse_args())

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default_2.xml')

name = args["name"]

if not os.path.exists('./dataset/' + name):
    os.makedirs('./dataset/' + name)

isopen = cv2.VideoCapture(0)
while (isopen.isOpened()):
    for i in range(1, 1000):
        ret, img = isopen.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            cv2.imshow("temp", roi_color)

            cv2.waitKey(100)
            cv2.imwrite("./dataset/" + name + "/pic.1." + str(i) + ".png", roi_color)
            print(i)
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # isopen.release()
        # cv2.destroyAllWindows()
        break
