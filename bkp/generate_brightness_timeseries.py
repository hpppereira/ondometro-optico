"""
Generate Time Series of Brightness

Henrique Pereira
2018/10/05
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import mlab
import cv2
from glob import glob
plt.close('all')

def generate_time_series(pathname, framei, framef):
    """
    generate brightness time series
    Input:
    - pathname - path a filename
    - framei - initial frame
    - framef - final frame
    """

    # read video
    cap = cv2.VideoCapture(pathname)
    # frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    # sample time
    dt = 1.0 / fps
    # time series of brightness
    st = []
    # loop for each frame
    for ff in range(framei,framef):
        print (ff)
        # read frame starting in ff
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
        ret, frame = cap.read()
        # take the RGB value for each point
        point = frame[pixloc[npoint][0],pixloc[npoint][1]]
        # calcula o brilho
        bri = np.sum(point[0]+point[1]+point[2])/3
        # append da serie de brilho
        st.append(bri)
    # time vector
    time = np.arange(framei*dt, framef*dt-dt, dt) # verificar
    # dataframe
    df = pd.DataFrame(st, index=time, columns=[npoint])
    # time as index
    df.index.name = 'time'
    return df, frame

def plot_points(frame, pixloc):
    """"
    plot image with points
    """
    plt.figure(figsize=(8,8))
    plt.imshow(frame)
    for p in pixloc.keys():
        plt.plot(pixloc[p][1],pixloc[p][0],'o')
        plt.text(pixloc[p][1],pixloc[p][0],p, color='w', fontsize=8)
        plt.show()
    return

def save_csv(pathout):
    """
    Save dataframe
    """
    df.to_csv(pathout)
    return

if __name__ == '__main__':

    pathname = glob(os.environ['HOME'] + '/Documents/ondometro_videos/'\
                    'laboceano/T100_570003_CAM1*')[0]
    print(pathname)

    pathout = '../out/brightness.csv'

    # initial and final frame to process
    framei = 500
    framef = 1500

    # dicionario de pontos a serem recuperados na imagem
    pixloc = {
              'p1': [750,500], #array horizontal
              'p2': [800,200],
              'p3': [800,300],
              'p4': [820,780], # ponto de iluminacao
              'p5': [900,1500], # slope array
              'p6': [950,1500],
              'p7': [900,1450],
              'p8': [950,1450],
              }

    # numero do ponto a ser extraida a serie temporal
    npoint = 'p1'

    # call functions
    df, frame = generate_time_series(pathname, framei, framef)
    plot_points(frame, pixloc)
    save_csv(pathout)
