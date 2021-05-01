#!/usr/bin/python

"""
Programa para reconhecer os digitos na imagem e salvar
os frames com o tempo no nome

Objetivo: Salvar frames sinconizados nas 2 filmagens

Dependencias:
- python 2.7
- sklearn 0.18.2
- skimage 0.13.0
- cv2 3.4.1

Metodologia:
- depois de salvar os tempos obtidos pelo OCR, fazer uma analise do intervalo
de tempo entre os tempos.
p ex: datetime_ocr[1]-datetime_ocr[0] = Timedelta('0 days 00:00:00.031000')
se for maior que 0.31, ou algo do tipo, o OCR falhou. nao salvar

Referencia
http://hanzratech.in/2015/02/24/handwritten-digit-recognition-using-opencv-sklearn-and-python.html
"""


# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import os
import pandas as pd

# os.system('rm /home/hp/GoogleDrive/Ondometro_Optico/out/OCR/teste2/*')

# stop
# cv2.destroyAllWindows()
plt.close('all')

filename = 'T100_570003_CAM3.avi'
pathname = os.environ['HOME'] + '/Documents/ondometro_data/laboceano/'
# pathfig = os.environ['HOME'] + '/Documents/ondometro_results/{}/'.format(filename)

# pathname of frames
# pathname = os.environ['HOME'] + '/Documents/ondometro_videos/CAMERA 1/T100/'

# image to be used
# filename = 'T100_520002_CAM1.avi'
# filename = os.listdir(pathname)[0]

print (filename)
# stop

# Get the path of the training set
# parser = ap.ArgumentParser()
# parser.add_argument("-c", "--classiferPath", help="Path to Classifier File", required="True")
# parser.add_argument("-i", "--image", help="Path to Image", required="True")
# args = vars(parser.parse_args())

# Load the classifier
clf, pp = joblib.load('digits_cls.pkl')

cap = cv2.VideoCapture(pathname + filename)

#contador de frames
f = -1

# tempo retirado da imagem em string
time_ocr_str = []

# for ff in range(0,40):
while(cap.isOpened()):

    print ('a')

    f += 1
    # if f > 10:
    #     break

    # cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

    ret, frame = cap.read()

    if frame is None:
        break
    else:

        print ('ret: {}'.format(ret))


        # Read the input image
        # im = cv2.imread('photo_2.jpg')
        # im1 = cv2.imread('ii_20160316_103003840_096.bmp')


        im1 = np.copy(frame)

        # stop

        # im = np.copy(im1[954-100:998+100,50-30:308+100,:])
        im = np.copy(im1[954:998,50:308,:])

        im11 = np.copy(im)
        im111 = np.copy(im11)


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
        imm, ctrss, hier = cv2.findContours(im_th.copy(),
                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        #cria lista com tempo (HH:MM:SS.SSS)
        list_nbr = []

        cont = -1
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

            # region of interesting (retangulo do numero)
            roi1 = im_th[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

            # Resize the image
            roi2 = cv2.resize(roi1, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi2, (3, 3))

            # Calculate the HOG (histogram of oriented gradients) features
            roi_hog_fd1 = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
            roi_hog_fd = pp.transform(np.array([roi_hog_fd1], 'float64'))
            nbr = clf.predict(roi_hog_fd)

            ###############################################################################
            # retira ambiguidades

            # tira ambiguidade de 0
            if nbr == 0:

                # break

                # 0 --> 9
                if roi[16,int(roi.shape[1]/2)] > 200 and roi[18,3] < 100:
                    nbr = np.array([9])

                # 0 --> 3 roi[13,6:20
                if (roi[13,:4] == 0).all():# and roi[11,roi.shape[1]] < 100:
                    nbr = np.array([3])


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
                    # stop

                # 2 --> 6
                if roi[13,3] == 255:
                    nbr = np.array([6])
                    # stop

            # 5 --> 6
            if nbr == 5:
                if roi[22,4] == 255:# and roi[11,roi.shape[1]] < 100:
                    nbr = np.array([6])

            # 6 --> 9
            if nbr == 6:
                if roi[18,2] == 0:# and roi[11,roi.shape[1]] < 100:
                    nbr = np.array([9])

            # 9 --> 5
            if nbr == 9:
                # stop
                if roi[6,24] == 0:# and roi[11,roi.shape[1]] < 100:
                    nbr = np.array([5])
                    # stop

            # stop




            #####################################3#########################################

                # stop
                # print f
                # print 'aaaaa'
                # break


            # cv2.putText(im11, str(int(nbr[0])), (rect[0]-4, rect[1]+35),cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(im11, str(int(nbr[0])), (rect[0]-4, rect[1]+35),cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)

            list_nbr.append(int(nbr))

        time_ocr_str.append('%s:%s%s:%s%s.%s%s%s' %tuple(list_nbr))

        print (time_ocr_str[-1])

        # datetime_ocr = pd.to_datetime(time_ocr_str)

        # if len(datetime_ocr) > 2:

            # dif = datetime_ocr[-1]-datetime_ocr[-2]

            # if dif < pd.Timedelta('0 days 00:00:00.031000'):

                # print time_ocr_str[-1] + ' -- ' + str(dif)

        plt.imshow(im11)

        cv2.imwrite(os.environ['HOME'] + '/Documents/ondometro_data/ocr/CAM3/%s_%s_frame_%s.png' %(filename[:-4], time_ocr_str[-1], f), im1)

        # stop

        # if f > 2:
        #     break
        # if list_nbr == [0, 0, 0, 0, 0, 0, 3, 0]:
        #     stop

# datetime_ocr = pd.to_datetime(time_ocr_str)

    # stop



# cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
# cv2.imshow("Resulting Image with Rectangular ROIs", im11)
# cv2.waitKey()
