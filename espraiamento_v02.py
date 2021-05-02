"""
calculo do fluxo optico

V = (dL/dT) / (dL/dX)
"""

# bibliotecas
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

pathname = os.environ['HOME'] + \
           '/Documents/ondometro/data/praia_seca_20180405/Cel_Nelson/filme_02/frames/'

filename1 = 'frame_00:03:36.000000_001.png'
filename2 = 'frame_00:03:36.030000_002.png'

# leitura dos dados
im1 = cv2.imread(pathname + filename1)
im2 = cv2.imread(pathname + filename2)

# converte de  uint8 para float32
# im1 = im1.astype(np.float32)
# im2 = im2.astype(np.float32)

# converte para escala de cinza
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# alisamento da imagem
im1 = cv2.GaussianBlur(im1, (5, 5), 0)
im2 = cv2.GaussianBlur(im2, (5, 5), 0)

# threshold
im1 = cv2.adaptiveThreshold(im1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,21,6)

im2 = cv2.adaptiveThreshold(im2,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,21,6)


# derivada espacial (coluna)
d = np.zeros(im1.shape)
for c in range(im1.shape[1]):
    d[:-1,c] = np.diff(im1[:,c])

# derivada temporal (frame posterior menos anterior)
im = im2 - im1

# calculo da velocidade (fluxo optico?)
v = im / d

# arctan v (colocar valores entre 0 e 255)
s = (np.arctan(v)/(np.pi/2) + 1) * (255/2)

# retira valores espurios
# v[np.where(v>30)] = 30
# v[np.where(v<-30)] = -30


# stop

# classificacao do v
# v[np.where(np.isnan(v))] = 0
# v[np.where(v>0)] = 100
# v[np.where(v<0)] = -100


plt.figure()
plt.imshow(v, cmap='RdBu')
plt.colorbar()

########################################
# tentativa de  criar contornos das derivadas das colunas

plt.figure()
plt.imshow(im1)
for cc in np.arange(0,1300,100):
    plt.plot(d[:,cc]*3+cc, range(len(d)),'-')


plt.figure()
plt.imshow(v, cmap='RdBu')
for cc in np.arange(0,1300,100):
    plt.plot(v[:,cc]+cc, range(len(d)),'-')
plt.colorbar()

plt.show()


    # d.append(np.diff(im1[:,c]))

# dd = np.array(d).T


# plotagem
# cv2.imshow('a', im1)