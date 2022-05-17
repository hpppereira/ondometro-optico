"""
Programa para deteccao do nivel de espraiaamento e esmaramento

Procedimentos:
- tetectar a primmeira derivada de baixo para cima

Referencias:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
https://docs.opencv.org/3.3.0/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html
https://docs.opencv.org/trunk/db/d8e/tutorial_threshold.html
"""

# teste tracking

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

cv2.destroyAllWindows()
plt.close('all')

pathname = os.environ['HOME'] + \
           '/Documents/ondometro/data/praia_seca_20180405/Cel_Nelson/filme_02/'

filename = '20180405_165638.mp4'

cap = cv2.VideoCapture(pathname + filename)

# inicia a filmagem no frame x

# divide por 2 a resolucao da camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)

ret, frame = cap.read()
# ret, frame1 = cap.read() #pega o frame seguinte

# cria variavel 'im' que sera manipulada
img = np.copy(frame)

# rotaciona imagem
img = cv2.flip(img, 0)

# converte para cinza
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# suaviza a imagem
# im = cv2.GaussianBlur(im, (5, 5), 0)

# ---------------------------------------------------------------------- #
# acha o nivel de espraiamento e esmaramento

# varia as colunas
# # cont = 0
# for c in np.arange(im.shape[1]):

#     # cria vetor coluna com alisamento
#     vc = pd.rolling_mean(im[:,c], 5)

#     # calcula a primeira derivada da coluna
#     d1 = np.diff(vc)
#     d2 = np.diff(d1)
#     sd = d1[:-1] + d2


    # plt.figure()
    # plt.imshow(im)
    # plt.plot(vc, range(len(vc)))
    # plt.plot(d1*10, range(len(vc)-1))
    # plt.plot(d2*10, range(len(vc)-2))
    # plt.plot(sd*10, range(len(vc)-2))
    # plt.show()

    # stop

    # # cont += 1

    # # janela da  media movel
    # win = 10

    # # vetor coluna
    # vc = pd.rolling_mean(im_blur[a:b,c], win)#[win-1:]
    # # vc = im_blur[a:b,c]

    # # acha picos
    # # peakind = signal.find_peaks_cwt(vc, np.arange(1,120))


    # # acha a primeira crista
    # ind_crista.append(np.where(vc == np.nanmax(vc))[0][0])

    # # se o limite ultrapassar um valor (quer dizer,  achou outra crista),
    # # entao repete o valor anterior

    # if len(ind_crista)>1:
    #     if np.abs(ind_crista[-1] - ind_crista[-2]) > 10:
    #         # print 'erro'
    #         ind_crista[-1] = ind_crista[-2]


# ---------------------------------------------------------------------- #
# simple threshold

# lim = 125
# ret,thresh1 = cv2.threshold(img,lim,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,lim,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,lim,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,lim,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,lim,255,cv2.THRESH_TOZERO_INV)
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [im, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in xrange(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.show()

# stop

# ---------------------------------------------------------------------- #
# Adaptive Thresholding

plt.figure()

# img = cv2.GaussianBlur(img,(5,5),0)
img = cv2.medianBlur(img,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,21,6)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,21,4)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# ---------------------------------------------------------------------- #
# Otsu's Binarization


plt.figure()

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()


# ---------------------------------------------------------------------- #
# plotagem

# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('image', width/2, height/2)
# cv2.imshow('image',im)


















# while(cap.isOpened()):

#     ret, frame = cap.read()

#     # muda resolucao
#     # frame = cv2.resize(frame, (width/2, height/2))

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # cv2.imshow('frame',gray)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break


cap.release()
# cv2.destroyAllWindows()
# plt.show()