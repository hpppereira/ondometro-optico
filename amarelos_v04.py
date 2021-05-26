"""
Programa para achar pontos amarelos nas cristas

24/10/2017

pixels da camera
(1080,1920)
"""

import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy import signal

plt.close('all')

# =============================================================================================== #
# Input Files

# path do video
pathname = os.environ['HOME'] + '/Dropbox/ondometro/data/videos/CAM_01/T100/frames/'

# regiao de interesse
a, b = 650, 950

# linha que identifica o inicio da agua
# linha_agua = 650

# plot_figure = 0

# acha os limites onde deve ser procurado por cristas (excluir batedor)
# o y=0 eh na parte superior
# xi = 0
# xf = 1920
# yi = 400
# yf = 1080


# cria lista com nome dos arquivos
listfiles = []
for arq in np.sort(os.listdir(pathname)):
	if arq.endswith('.png'):
		listfiles.append(arq)


image = cv2.imread(pathname + listfiles[150])
# im2 = cv2.imread(pathname + listfiles[151])

# resolucao da imagem
ll, cc = image.shape[:2]

# converte para escala de cinza
im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# suaviza a imagem
im_blur = cv2.GaussianBlur(im_gray, (11, 11), 0)

# limites da imagem
# ret, im_th = cv2.threshold(im_gray, 50, 80, cv2.THRESH_BINARY_INV)


# loop para identificacao de contraste


# -------------------------------------------------------------------- #
# plotagem das linhas de valores em pixel por coluna

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111)
# ax1.imshow(image)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(image)


ind_crista = []

# varia as colunas
cont = 0
for c in np.arange(0,cc,1):

	cont += 1

	# janela da  media movel
	win = 10

	# vetor coluna
	vc = pd.rolling_mean(im_blur[a:b,c], win)#[win-1:]
	# vc = im_blur[a:b,c]

	# acha picos
	# peakind = signal.find_peaks_cwt(vc, np.arange(1,120))


	# acha a primeira crista
	ind_crista.append(np.where(vc == np.nanmax(vc))[0][0])

	# se o limite ultrapassar um valor (quer dizer,  achou outra crista),
	# entao repete o valor anterior

	if len(ind_crista)>1:
		if np.abs(ind_crista[-1] - ind_crista[-2]) > 10:
			# print 'erro'
			ind_crista[-1] = ind_crista[-2]




	# print ind_crista

	# vetor de primeira derivada
	# v1 = np.diff(vc)

	# acha valores positivos
	# v2 = np.sign(v1)
	
	# acha picos
	# peakind = signal.find_peaks_cwt(vc, np.arange(1,120))

	# acha os indices dos picos dentro da agua
	# pkid = np.where((peakind>a) & (peakind<b))[0]

	# 

	# plt.figure()
	# plt.plot(range(len(vc)),vc, '-b', peakind, vc[peakind],'ro')

	# plt.show()
	# stop




	# ax1.plot(im_gray[a:b,c] + c,range(a,b))
	# ax2.plot(im_blur[a:b,c] + c,range(a,b))
	# ax2.plot(vc + c,range(a,b))


plt.plot(range(len(ind_crista)), np.array(ind_crista)+a,'r')
plt.show()

# Threshold the image
# ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
# ret, im_th = cv2.threshold(im_gray, 18, 60, cv2.THRESH_BINARY_INV)


# plt.figure()
# plt.imshow(im_gray)

# plt.figure()
# plt.imshow(im_blur)


# plt.show()
# Find contours in the image
# im, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# stop


# im = im2 - im1

# im = np.copy(im1[600:,:])

# plotar com cv2.imshow no tamanho da tela
# height, width = im.shape[:2]

# cv2.namedWindow('a',  cv2.WINDOW_NORMAL)
# cv2.resizeWindow('a', width, height)
# cv2.imshow('a',  im)
# r = cv2.waitKey(-1)
# print 'DEBUG: waitKey returned:',  chr(r)
# cv2.destroyAllWindows()



















# # tamanho da janela na media movel da derivada da coluna

# for f in listfiles[103:104]:

# 	print f

# 	im1 = cv2.imread(pathname + f)
# 	im = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)


# 	amarelos = []

# 	# varia as colunas
# 	for c in range(im.shape[1]):

# 		# modulo da derivada da coluna
# 		dc = np.abs(np.diff(im[:,c]))

# 		# media movel
# 		mm = pd.rolling_mean(dc, 30)

# 		# normaliza
# 		mmn = mm / np.nanmax(mm)

# 		# acha a posicao dos maximas derivadas
# 		# pmax = np.argsort(mm)[-nmax:]
# 		pmax = np.where(mmn>0.7)[0]

# 		amarelos.append(pmax)

# 		plt.plot(np.ones(len(amarelos[c]))*c, amarelos[c],'y.', markersize=0.03)


# 	# plt.savefig('../fig/amarelos3_%s.png' %listfiles[a],  dpi=100)

# plt.imshow(im1)





# # if plot_figure == 1:

# # 	plt.figure()
# # 	plt.imshow(img1)

# # 	plt.figure()
# # 	plt.contour(np.flipud(img1))

# # 	plt.figure()
# # 	plt.plot(mm)

# # 	plt.figure()
# # 	plt.plot(mm)
# # 	plt.plot(pmax, np.ones(len(pmax)),'o')



# plt.show()