"""
distancia focal em pixels = 1250 (equivale a  59 cm)

CRISTA

camera superior / linha horizonte: 120
camera superior / crista: 220
angulo entre horizonte e crista: np.arctan(100.0/1250)*180/pi

camera inferior / linha horizonte: 306
camera inferior / crista: 360
angulo entre horizonte e crista: np.arctan(54.0/1250)*180/pi

ESPRAIAMENTO

camera superior / linha horizonte: 136
camera superior / espraiamento: 557 
angulo entre horizonte e crista: arctan((557-136)/1250.)*180/pi

camera inferior / linha horizonte: 317
camera inferior / espraiamento: 564
angulo entre horizonte e crista: np.arctan(54.0/1250)*180/pi



"""

# importa bibliotecas
import os
import matplotlib.pyplot as plt
import cv2

plt.close('all')

pathname_inf = os.environ['HOME'] + '/Documents/ondometro/data/praia_seca_20180405/Cel_Victor/filme_02/frames/'
pathname_sup = os.environ['HOME'] + '/Documents/ondometro/data/praia_seca_20180405/Cel_Nelson/filme_02/frames/'

# leitura imagem infeior (cam victor)

# crista superior
cr_inf = cv2.imread(pathname_inf + 'frame_00:01:41.420000_015.png')

# espraiamento
es_inf = cv2.imread(pathname_inf + 'frame_00:01:35.000000_001.png')
    
# leitura imagem superior (cam nelson)

# crista
cr_sup = cv2.imread(pathname_sup + 'frame_00:03:36.720000_025.png')

# espraiamento
es_sup = cv2.imread(pathname_sup + 'frame_00:03:30.270000_010.png')


plt.figure()
plt.imshow(es_inf)

plt.figure()
plt.imshow(es_sup)

plt.show()