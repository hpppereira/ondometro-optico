"""
Descricao:
Calculo da altura de onda por estereoscopia

Projeto:
- Ondometro Optico

Data: 18/05/2018

Infos:
- Utilizar como base desenho no arquivo 'xxx.png' (matheus)
- alfa - angulo horizonte as cristas
  beta - angulo horizonte a nivel medio

Obervacoes
1 - Neste script estou inserindo os dados manualmente.
aqui as coordenadas dos pixels referentes à estes
pontos deverá ser detectada automaticamente.

2 - Triângulos semelhantes:
Cada câmera tem 2 triângulos semelhantes, logo, dois catetos adjacentes
e tem o cateto oposto de valor igual. Podemos calcular 2 ângulos alpha para
a câmera superior e 2 ângulos beta para a câmera inferior.

3 - Cálculo da altura da onda:
Utilizando métodos trigonométricos podemos calcular a
altura da onda após calcular os comprimentos s1 e s2.
Temos 2 comprimentos horizontais (x1, x2) que equiva-
lem aos catetos opostos dos triângulos e são semelhantes
à distância focal.
x1 --> das câmeras até a crista.
x2 --> das câmeras até o ponto de referência.

Calculos
dist_focal: 1250

- posicao vertical do pixel (camera superior)
* horizonte: 120
* crista: 220
angulo entre horizonte e crista:
np.arctan((crista - horizonte)/dist_focal)*180/pi

- posicao vertical do pixel (camera inferior)
* horizonte: 306
* crista: 360
angulo entre horizonte e crista:
np.arctan((crista - horizonte)/dist_focal)*180/pi
np.arctan(54.0/1250)*180/pi

- posicao vertical do pixel (camera superior)
* horizonte: 136
* nivel_medio: 557
angulo entre horizonte e nivel medio:
np.arctan((nivel_medio - horizonte)/dist_focal)*180/pi
arctan((557-136)/1250.)*180/pi

- posicao vertical do pixel (camera inferior)
* horizonte: 317
* nivel_medio: 557
angulo entre horizonte e nivel medio:
np.arctan((nivel_medio - horizonte)/dist_focal)*180/pi
np.arctan(54.0/1250)*180/pi
"""

###############################################################################
# importa bibliotecas

import numpy as np


###############################################################################
# Dados de entrada

# distancia vertical entre as cameras (metros)
dist_mt_cam = 2.04

# distância focal da câmera (entrara aqui, um parametro de calibração da camera
# para calcular a distancia focal)
dist_px_focal = 1250

# posicao do horizonte superior e inferior (frame 1 e 2?)
# hor_sup_1 = 120
# hor_sup_2 = 136
# hor_inf_1 = 306
# hor_inf_2 = 317

# chute inicial para a posicao x,y do nivel medio
pos_ho_sup = 120
pos_ho_inf = 306

# Posição da crista na câmera superior e inferior
pos_cr_sup = 220
pos_cr_inf = 360

# Posicao do nivel medio (ponto de referencia) camera superior e inferior
pos_nm_sup = 557
pos_nm_inf = 564

###############################################################################
# inicio dos calculos

# -----------------------------------------------------------------------------
# camera superior

# Calculo do cateto oposto da camera superior e inferior referente a crista

# dos ângulos alpha da câmera superior:
# co_sup_1 = cristasup - hsup1
# cat_sup_2 = rpsup - hsup2

co_cr_sup = pos_cr_sup - pos_ho_sup
co_cr_inf = pos_cr_inf - pos_ho_inf

# cateto oposto referente ao nivel medio
co_nm_sup = pos_nm_sup - pos_ho_sup
co_nm_inf = pos_nm_inf - pos_ho_inf

# angulo entre horizonte e crista
ang_cr_sup = np.arctan(co_cr_sup / dist_px_focal)
ang_cr_inf = np.arctan(co_cr_inf / dist_px_focal)

# angulo entre horizonte e nivel medio (ponto de referencia)
ang_nm_sup = np.arctan(co_nm_sup / dist_px_focal)
ang_nm_inf = np.arctan(co_nm_inf / dist_px_focal)


# -----------------------------------------------------------------------------
# camera inferior

# #Cálculo dos angulos beta da câmera inferior:
# catinf1 = cristainf - hinf1
# catinf2 = RPinf - hinf2
#
# # ???
# beta1 = math.degrees(math.atan(catinf1 / FD))
# beta2 = math.degrees(math.atan(catinf2 / FD))

# distncia horizontal ate a crista
# dist_ho_cr = dist_mt_cam / (np.tan(ang_cr_sup) - np.tan(ang_cr_inf))

# distncia horizontal ate o nivel medio
x = dist_mt_cam / (np.tan(ang_cr_sup) - np.tan(ang_cr_inf))

# x calculado pelo victor
x1 = 25.5
x2 = 5.86

s1 = x * ang_cr_sup
s2 = (np.tan(ang_cr_inf) * (dist_mt_cam/x)) * x


# z = dist_mt_cam * np.tan(ang_cr_inf) / (np.tan(ang_cr_sup) - np.tan(ang_cr_inf))

# x1 = dist_px_focal / ()
# ??
x1 = dist_mt_cam / (math.tan(ang_cr_sup) - math.tan(ang_cr_inf))

# ??
x2 = DH / (math.tan(math.radians(alpha2)) -
               math.tan(math.radians(beta2)))

# ??
s1 = (math.tan(math.radians(beta1))) * x1
s2 = (math.tan(math.radians(beta2))) * x2

# ??
WH = s2 - s1



# print('A altura da onda é de %f cm' % round(WH, ndigits=2))
# print('A altura da onda é: ', round(WH, ndigits=3), 'cm')
