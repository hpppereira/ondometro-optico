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

plt.close('all')



# =============================================================================================== #
# Input Files

# path do video
pathname = os.environ['HOME'] + '/Dropbox/Ondometro_Optico/data/videos/CAMERA\ 01/T100/frames/'

# lista arquivos dentro do diretorio
# list_frames = np.sort(os.listdir(pathname))

# im = cv2.imread(pathname + list_frames[30])


# =============================================================================================== #
# Functions

# leitura do video
# cap = cv2.VideoCapture(pathname + filename)

# leitura do frame
# ret, frame = cap.read(100000)

# pega resolucao da camera
# res = frame.shape

# rotationa frame
# frame = ndimage.rotate(frame, -90)

# stop
# tamanho da janela na media movel da derivada da coluna
win = 30

# numero de pontos maximos para pegar na coluna
nmax = 150

plot_figure = 0

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

# stop
# imagem do primeiro frame, que sera subtraido dos outros para remover ruido

for a in range(len(listfiles)-1):


	## flatten=0 if image is required as it is 
	## flatten=1 to flatten the color layers into a single gray-scale layer

	img = misc.imread(os.path.join(pathname, listfiles[a]), flatten = 0)
	img_fundo = misc.imread(os.path.join(pathname, listfiles[a+1]), flatten = 0)

	# plota imagem para ser sobreprosta as linhas
	plt.figure()
	plt.imshow(img)

	# imagem filtrada
	img1 = img_fundo[:,:,0] - img[:,:,0]

	# eh invertido x-y
	# img1 = img11[yi:yf,xi:xf,0]
	# img2 = img11[yi:yf,xi:xf,:]

	amarelos = []

	# varia as colunas
	for c in range(img1.shape[1]):

		# modula da derivada da coluna
		dc = np.abs(np.diff(img1[:,c]))

		# media movel
		mm = pd.rolling_mean(dc, win)

		# normaliza
		mmn = mm / np.nanmax(mm)

		# acha a posicao dos maximas derivadas
		# pmax = np.argsort(mm)[-nmax:]
		pmax = np.where(mmn>0.7)[0]

		amarelos.append(pmax)

		plt.plot(np.ones(len(amarelos[c]))*c, amarelos[c],'y.', markersize=0.03)

	plt.savefig('../fig/amarelos2_%s.png' %listfiles[a],  dpi=100)






# if plot_figure == 1:

# 	plt.figure()
# 	plt.imshow(img1)

# 	plt.figure()
# 	plt.contour(np.flipud(img1))

# 	plt.figure()
# 	plt.plot(mm)

# 	plt.figure()
# 	plt.plot(mm)
# 	plt.plot(pmax, np.ones(len(pmax)),'o')



# plt.show()