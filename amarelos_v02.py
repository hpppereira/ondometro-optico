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
pathname = os.environ['HOME'] + '/Dropbox/ondometro/data/videos/CAM_01/T100/frames/'


# numero de pontos maximos para pegar na coluna
nmax = 150

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


# tamanho da janela na media movel da derivada da coluna

for f in listfiles[103:104]:

	print f

	im1 = cv2.imread(pathname + f)
	im = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)


	amarelos = []

	# varia as colunas
	for c in range(im.shape[1]):

		# modulo da derivada da coluna
		dc = np.abs(np.diff(im[:,c]))

		# media movel
		mm = pd.rolling_mean(dc, 30)

		# normaliza
		mmn = mm / np.nanmax(mm)

		# acha a posicao dos maximas derivadas
		# pmax = np.argsort(mm)[-nmax:]
		pmax = np.where(mmn>0.7)[0]

		amarelos.append(pmax)

		plt.plot(np.ones(len(amarelos[c]))*c, amarelos[c],'y.', markersize=0.03)


	# plt.savefig('../fig/amarelos3_%s.png' %listfiles[a],  dpi=100)

plt.imshow(im1)





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



plt.show()