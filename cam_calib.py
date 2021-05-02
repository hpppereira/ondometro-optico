# -*- coding: utf-8 -*-
"""
Programa para calubracao da camera
- pega caracteristicas de:
resolucao, angulo de abertura...

Filmagem para obetanção dos parâmetros:
20180316_152832.mp4
Dimensões dos livros:
29,5cm x 20,9cm
O celular  estava na vertical e apoiado sobre o livro mais próximo.
Portanto distava de 2 x 29,5cm do livro filmado de pé. 
A câmera fica a 9,8cm do pé do celular.
A pergunta é:
Qual é o campo? ( X graus x Y graus ?)

# mudar rgb
# frame[:,:,1] = 255

"""


import os
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np

plt.close('all')

pathname = os.environ['HOME'] + '/Dropbox/ondometro/data/preproc/'
filename = '20180316_152832.mp4'


#######################################################3
# parameros de entrada

# altura da camera
alt_cam = 9.8

#dimensao do objeto horizontal
comp_h_cm = 20.9

#dimensao do objeto vertical
comp_v_cm = 29.5

# distancia da lente para o objeto
dist_obj = 59

# retangulo do livro
    			  #a  b    c     d
obj = np.array([[115, 550, 550, 115],
				[732, 732, 107, 107]]).T

#######################################################3



#######################################################3

# leitura do video
cap = cv2.VideoCapture(pathname + filename)

# leitura do frame
ret, frame = cap.read(10)

# pega resolucao da camera
res = frame.shape

# rotationa frame
frame = ndimage.rotate(frame, -90)

# vetor horizontal e vertical das dimensoes do livro
vet_h_px = np.arange(obj[0,0], obj[1,0])
vet_v_px = np.arange(obj[2,1], obj[0,1])

# comprimento do objeto em píxels
comp_h_px = len(vet_h_px)
comp_v_px = len(vet_v_px)

# valor do pixel unitario horizontal
px_h_unit = comp_h_cm / comp_h_px
px_v_unit = comp_v_cm / comp_v_px

# relacao entre px horizontal e vertical
px_hv_rel = px_h_unit / px_v_unit

#distancia horizontal em pixels
# dist_h_px = px_h_unit * 

#######################################################
# calculo do angulo de abertura vertical

# triangulo - A

# hipotenusa superior
hip_a = np.sqrt((comp_v_cm-alt_cam)**2 + dist_obj**2)

# angulo do triangulo superior - A
ang_a = np.rad2deg(np.arctan((comp_v_cm-alt_cam) / dist_obj))

# triangulo - B

# hipotenusa superior
hip_b = np.sqrt(alt_cam**2 + dist_obj**2)

# angulo do triangulo superior - B
ang_b = np.rad2deg(np.arctan(alt_cam / dist_obj))

# angulo de abertura vertical da camera
ang_v = ang_a + ang_b

#######################################################
# calculo do angulo de abertura horizontal
## supondo que a camera esta no centro horizontal do
## objeto

# triangulo - C

# hipotenusa superior
hip_c = np.sqrt((comp_h_cm/2)**2 + dist_obj**2)

# angulo do triangulo superior - C
ang_c = np.rad2deg(np.arctan((comp_h_cm/2) / dist_obj))

# angulo de abertura vertical da camera
ang_h = 2 * ang_c


#######################################################
# calcula valor de 1 px em centimetros em funcao da
# distancia
# ps: quanto maior a distancia maior a distancia que  
# 1 px representa

# distancia da lente a ser calculado o valor em cm de 1 px
n_dist = 100

# nova 
n_dist_v = (np.tan(np.deg2rad(ang_v/2)) * n_dist) * 2
n_dist_h = (np.tan(np.deg2rad(ang_h/2)) * n_dist) * 2

# valor em cm do pixel na vertical
n_px_v = n_dist_v / comp_v_px
n_px_h = n_dist_h / comp_h_px

# valor do novo pixel horizontal calculado pela relacao
# entre pixels
n_px_h1 = n_px_v * px_hv_rel

#######################################################
# figura

# desenha retangulo na imagem
cv2.rectangle(frame, (vet_h_px[0],vet_v_px[-1]), (vet_h_px[-1], vet_v_px[0]),  (255,0,0), thickness=5)

# desenha distancias horizonais e veritcais
# horizontal
cv2.arrowedLine(frame, (vet_h_px[0], 400), (vet_h_px[-1], 400),  (0,0, 255), thickness=5)
cv2.arrowedLine(frame, (vet_h_px[-1], 400), (vet_h_px[0], 400),  (0,0, 255), thickness=5)

#vertical
cv2.arrowedLine(frame, (330, vet_v_px[0]), (330, vet_v_px[-1]),  (0,0, 0), thickness=5)
cv2.arrowedLine(frame, (330, vet_v_px[-1]), (330, vet_v_px[0]),  (0,0, 0), thickness=5)

# escreve valores de distancia e pixel
# cv2.putText(frame,"Hello World!!!", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, 
# cv2.putText(frame, u'Dist: %s' %, (130, 451), 0, 1, (0, 0, 255), 2, cv2.LINE_AA)

plt.imshow(frame)

plt.show()


