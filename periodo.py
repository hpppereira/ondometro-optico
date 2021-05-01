"""
Calculo do periodo de onda
"""

import cv2
# from sklearn.externals import joblib
# from skimage.feature import hog
import numpy as np
# import argparse as ap
import matplotlib.pyplot as plt
import os
import pandas as pd
# import pandas as pd
from matplotlib import mlab


plt.close('all')

###############################################################################

# pathname of frames
pathname = os.environ['HOME'] + '/Documents/GoogleDrive_old/Ondometro_Optico/data/videos/CAMERA 1/T100/'

# image to be used
filename = 'T100_110000_CAM1.avi'

cap = cv2.VideoCapture(pathname + filename)
ret, frame = cap.read()
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/fps

# dicionario de pontos a serem recuperados na imagem
points = {'p01': [800,100], #array horizontal
          'p02': [800,200],
          'p03': [800,300],
          'p04': [900,750], # ponto de iluminacao
          'p05': [900,1500], # slope array
          'p06': [950,1500],
          'p07': [900,1450],
          'p08': [950,1450]}


# cria arquivo de brilho
# gera_brilho = False
gera_brilho = False

# frame inicial e final para processar
framei = 500
framef = 1000 # length

# comprimento do video (numero total de frames)
time = np.arange(framei*dt,framef*dt,dt) # verificar

###############################################################################
# abre arquivo de brilho

if gera_brilho == False:

    df = pd.read_csv('../out/brilho.csv', index_col='time')

###############################################################################
# gera arquivo de brilho de um ponto na imagem

if gera_brilho == True:

    st = [] #serie temporal de tons de cinza
    for ff in range(framei,framef):

        print ff
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

        ret, frame = cap.read()

        # stop

        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculo do brilho
        point = frame[800,1500]

        bri = np.sum(point[0]+point[1]+point[2])/3 

        # //Luminosity := (299*RGB.rgbtred+587*RGB.rgbtgreen+114*RGB.rgbtblue) DIV 1000;
        # bri = (299*point[2]+587*point[1]+114*point[0]) / 1000 

        st.append(bri)

    df = pd.DataFrame(st, index=time, columns=['point_01'])
    df.index.name = 'time'

    df.to_csv('../out/brilho.csv')


###############################################################################
# plotagem

plt.imshow(frame)
# plotagem dos pontos de medicao
for p in points.keys():

    x = points[p][1]
    y = points[p][0]

    plt.plot(x,y,'o')
    plt.text(x,y,p, color='w', fontsize=10)

plt.show()


###############################################################################
# calculo do periodo de onda



# pega o tempo especifico de onde a onda ja chegou
df = df[20:]

# calculo do espectro
nfft = len(df)/2
fs = fps

spec = mlab.psd(df.point_01,NFFT=nfft,Fs=fs,detrend=mlab.detrend_mean,
              window=mlab.window_hanning,noverlap=nfft/2)

f, sp = spec[1][1:],spec[0][1:]







# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.imshow(frame)
