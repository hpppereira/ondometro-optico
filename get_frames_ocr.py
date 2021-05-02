#!/usr/bin/python

# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import os

cv2.destroyAllWindows()
plt.close('all')

# pathname of frames
pathname = os.environ['HOME'] + '/Dropbox/ondometro/data/videos/CAM_01/T100/'

# image to be used
filename = 'T100_570003_CAM1.avi'

# Get the path of the training set
# parser = ap.ArgumentParser()
# parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
# parser.add_argument("-i", "--image", help="Path to Image", required="True")
# args = vars(parser.parse_args())

# Load the classifier
clf, pp = joblib.load('digits_cls.pkl')

cap = cv2.VideoCapture(pathname + filename)

f = -1 #contador de frames
while(cap.isOpened()):
    
    f += 1
    ret, frame = cap.read()

    if ret:


        # Read the input image 
        # im = cv2.imread('photo_2.jpg')
        # im1 = cv2.imread('ii_20160316_103003840_096.bmp')

        im1 = np.copy(frame)

        # stop

        # im = np.copy(im1[954-100:998+100,50-30:308+100,:])
        im = np.copy(im1[954:998,50:308,:])

        im11 = np.copy(im)
        im111 = np.copy(im11)

        # stop
        # Convert to grayscale and apply Gaussian filtering
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # edged = cv2.Canny(blurred, 50, 200, 255)

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
        # ret, im_th = cv2.threshold(edged, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # inverte cores
        im_th = im_th * -1
        im_th = im_th + 255
        im_th = im_th.astype(np.uint8)

        # Find contours in the image
        imm, ctrss, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # acha apenas contornos maiores que 10 pontos (para remover os .)
        ctrs = []
        for c in ctrss:
            if len(c) > 10:
                ctrs.append(c)

        # stop
        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        # For each rectangular region, calculate HOG features and predict
        # the digit using Linear SVM.
        cont = -1

        #cria lista com tempo (HH:MM:SS.SSS)
        list_nbr = []

        for rect in np.flipud(rects):
        # for rect in rects:

            cont += 1

            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 


            # Make the rectangular region around the digit
            # rect[3] -- altura do retangulo
            # leng = int(rect[3] * 3.)
            # pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            # pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            # roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
            # roi = np.copy(im_th)

            # roi = im_th[6:44,231:260] #retangulo mais da direita

            # stop
            roi = im_th[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]


            # Resize the image
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            # Calculate the HOG features
            roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
            nbr = clf.predict(roi_hog_fd)

            # tira ambiguidade de 0
            if nbr == 0:

                # break

                # 0 --> 9                
                if roi[16,roi.shape[1]/2] > 200 and roi[18,3] < 100:
                    nbr = np.array([9])

                # 0 --> 3 roi[13,6:20
                if (roi[13,:4] == 0).all():# and roi[11,roi.shape[1]] < 100:
                    nbr = np.array([3])
                    stop

            # tira ambiguidade entre de 7            
            if nbr == 7:

                # 7 --> 1
                if rect[2] < 15:
                    nbr = np.array([1])


            # tira ambiguidade do 3
            if nbr == 3:

                # break

                # 3 --> 6
                if roi[13,2] == 255:# and roi[11,roi.shape[1]] < 100:
                    nbr = np.array([6])


            # tira ambigguidade de 2
            if nbr == 2:

                # 2 --> 8
                if (roi[13,6:20] == 255).all(): #se tiver um  tracejado no meio (8)
                    nbr = np.array([8])
                    # break


            cv2.putText(im11, str(int(nbr[0])), (rect[0]-4, rect[1]+35),cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

            list_nbr.append(int(nbr))

            # print nbr

            # escolhe o frame para parar

        cv2.imwrite('../out/frames_ocr_CAM_01_T100/' + filename[:-4] + '_%s:%s%s:%s%s.%s%s%s.png' %tuple(list_nbr), im)

        # if f > 2:
        #     break
        # if list_nbr == [0, 0, 0, 0, 0, 0, 3, 0]:
        #     stop


    # stop



# cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
# cv2.imshow("Resulting Image with Rectangular ROIs", im11)
# cv2.waitKey()
