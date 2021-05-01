
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pandas as pd
import cv2
plt.close('all')



# carrega imagem
frame = cv2.imread('img/swell_01.jpg')

img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# B, G, R = frame.T
# brilho = (B + G + R) / 3
# img = brilho.T

c = img[:,0]

stop
# tamanho da janela na media movel da derivada da coluna
win = 5

# numero de pontos maximos para pegar na coluna
nmax = 150

# valor maximo da derivada (normalizada??)
max_der = 0.7

# acha os limites onde deve ser procurado por cristas (excluir batedor)
# o y=0 eh na parte superior
# xi = 0
# xf = 1920
# yi = 400
# yf = 1080

# imagem filtrada
# img1 = img_fundo[:,:,0] - img[:,:,0]

# eh invertido x-y
# img1 = img11[yi:yf,xi:xf,0]
# img2 = img11[yi:yf,xi:xf,:]

amarelos = []

# varia as colunas
for c in range(img.shape[1]):

	# modulo da derivada da coluna
	dc = np.abs(np.diff(img[:,c]))

	# media movel
	mm = pd.rolling_mean(dc, win)

	# normaliza
	mmn = mm / np.nanmax(mm)

	# acha a posicao dos maximas derivadas
	# pmax = np.argsort(mm)[-nmax:]
	pmax = np.where(mmn>0.7)[0]

	amarelos.append(pmax)

	# plt.plot(np.ones(len(amarelos[c]))*c, amarelos[c],'y.', markersize=0.03)

plt.savefig('linha_cristas.png',  dpi=100)


plt.figure()
plt.imshow(img)

plt.figure()
plt.contour(np.flipud(img))

plt.figure()
plt.plot(mm)

plt.figure()
plt.plot(mm)
plt.plot(pmax, np.ones(len(pmax)),'o')



plt.show()
