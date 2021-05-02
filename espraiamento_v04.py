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

# stop
# teste tracking

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

# cv2.destroyAllWindows()
plt.close('all')

pathname = os.environ['HOME'] + \
           '/Documents/ondometro_videos/praia_seca_20180405/Cel_Nelson/filme_02/'

filename = '20180405_165638.mp4'

cap = cv2.VideoCapture(pathname + filename)

# stop
# inicia a filmagem no frame x

# divide por 2 a resolucao da camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_FRAMES, 4290)

ret, frame = cap.read()
# ret, frame1 = cap.read() #pega o frame seguinte

# cria variavel 'im' que sera manipulada
im = np.copy(frame)

# rotaciona imagem
im = cv2.flip(im, 0)

# imagem orinal rotacionada
orig = im

# converte para cinza
im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# stop
# calcula o brilho
# im = im.sum(axis=2)/3

# im = im.astype(np.uint8)

# stop
# suaviza a imagem
im1 = cv2.GaussianBlur(im, (75, 75), 0)
im2 = cv2.GaussianBlur(im, (15, 15), 0)
im3 = cv2.GaussianBlur(im, (15, 15), 0)

# stop
# threshold (chose one)
ret, th1 = cv2.threshold(im1, 119, 255, cv2.THRESH_BINARY)
# ret, th3 = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
# ret, th2 = cv2.threshold(im, 125, 255, cv2.THRESH_BINARY_INV)
# ret, th2 = cv2.threshold(im, 120, 255,cv2.THRESH_TRUNC)
# ret, th2 = cv2.threshold(img,lim,255,cv2.THRESH_TOZERO)
# ret, th = cv2.threshold(im,lim,255,cv2.THRESH_TOZERO_INV)
ret, th3 = cv2.threshold(im3,130,255,cv2.THRESH_BINARY)
# ret, th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret, th2 = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# th2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,6)
# th2 = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,4)

# im1 = th1 # imagem para detectacao do espraiamento
# im2 = th2 # imagem para detectar o horizonte
# im3 = th3 # imagem para detectar as cristas

# ---------------------------------------------------------------------- #
# acha o nivel de espraiamento

# varia as colunas
# cont = 0

# plt.figure()
# plt.imshow(orig)
# plt.imshow(th3)
# plt.show()

espraio = [] # vetor dos indices em y do espraiamento
horizonte = []
cristas = []

for c in np.arange(0,im.shape[1]):

    # cria vetor coluna com alisamento
    vc1 = th1[:,c] # vetor para detectar espraio
    vc2 = th2[:,c] # vetor para detectar horizonte

    # vc = pd.rolling_mean(im[:,c], 5)

    # calcula a primeira derivada
    # espraio
    d1 = np.diff(vc1)
    d1 = np.concatenate(([0],d1))

    # horizonte
    d2 = np.diff(vc2)
    d2 = np.concatenate(([0],d2))

    # d2 = np.diff(d1)
    # sd = d1[:-1] + d2

    # stop

    idx1 = np.where(d1!=0)[0][-1] # indices do espraio
    idx2 = np.where(d2!=0)[0][-1] # indices do horizonte

    # cristas (procura entre os limites de crista e  espraiamento)
    vc3 = th3[idx2:idx1,c] # vetor para detectar horizonte
    d3 = np.diff(vc3)
    d3 = np.concatenate(([0],d3))

    idx3 = np.where(d3==255)[0] # indices das crisas
    if len(idx3) > 0:
        idx3 = idx3[0]
    else:
        idx3 = np.nan


    # cria vetores com os indices das colunas
    espraio.append(idx1)
    horizonte.append(idx2)
    cristas.append(idx3+idx2)


    # cria loop entre os indices do horizonte e espraiamento
    # vc3 = pd.DataFrame(im3[idx2:idx1,c])
    # vc3 = vc3.rolling(window=10, center=False).max()

    # # calcula a altura maxima de um trecho da coluna
    # h = [] # intensidade maxima dos pixels entre um segmento
    # seg = 50 # tamanho do segmento
    # for i in np.arange(0,len(vc3),seg):

    #     # trecho do segmento
    #     aux = vc3[i:i+seg]

    #     # valores maximos e minimos
    #     hmax = aux.max().values[0]
    #     hmin = aux.min().values[0]
    #     h.append(np.abs(hmax-hmin))


    # stop

    # stop
    # plt.plot(c, idx,'k.')
    # plt.plot(c, idx1,'k^')


    # plt.figure()
    # plt.imshow(th3)
    # plt.plot(vc3-vc3.mean()+c, range(idx2,idx1),'r')
    # # plt.plot(d3, range(len(vc3)))
    # # plt.plot(d2*10, range(len(vc)-2))
    # # plt.plot(sd*10, range(len(vc)-2))

    # plt.show()
    # stop

# plt.show()


# vetor de espraio
espraio = pd.DataFrame(espraio)

# stop

ww = 30
espraio = espraio.rolling(window=ww, center=False).mean()

# preenche valores com nan com o primeiro valor do espraio
espraio.iloc[:ww,0] = espraio.iloc[ww,0]

plt.figure()
plt.imshow(orig)
plt.plot(range(im.shape[1]),espraio,'k')
plt.plot(range(im.shape[1]),horizonte,'r')
plt.plot(range(im.shape[1]),cristas,'y')

plt.show()

    # plt.close('all')

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

# plt.figure()

# # img = cv2.GaussianBlur(img,(5,5),0)
# img = cv2.medianBlur(img,5)
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,21,6)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,21,4)
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in xrange(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()


# # ---------------------------------------------------------------------- #
# # Otsu's Binarization


# plt.figure()

# # global thresholding
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
#           'Original Noisy Image','Histogram',"Otsu's Thresholding",
#           'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
# for i in xrange(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()


# # ---------------------------------------------------------------------- #
# # plotagem

# # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('image', width/2, height/2)
# # cv2.imshow('image',im)


















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
