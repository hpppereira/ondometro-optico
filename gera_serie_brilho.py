# Gera series de brilho
#
# Ultimas modificacoes:
# Henrique Pereira - 17/05/2022

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
plt.close('all')

if __name__ == "__main__":

    # paths
    pth_video = '/home/hp/shared/ufrjdrive/ondometro_data/videos/CAMERA 1/T100/'
    pth_out = '/home/hp/gdrive/ondometro_optico/'

    fln = 'T100_570002_CAM1.avi'

    # carrega video
    cap = cv2.VideoCapture(pth_video + fln)

    # abre um frame
    ret, frame = cap.read()

    # comprimento do video
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # frames por segundos
    fps = cap.get(cv2.CAP_PROP_FPS)

    # intervalo de amostragem
    dt = 1. / fps

    # frame inicial e final para processar
    framei = 1000
    framef = length #2024

    # ponto de medicao
    p = (830, 1230)

    # serie de tempo a ser processada
    time = np.arange(framei*dt, framef*dt, dt)

    # # plota figura com ponto da serie de bilho
    # fig, ax = plt.subplots()
    # ax.imshow(frame)
    # ax.plot(p[1], p[0], 'o')
    # plt.show()

    # inicia loop para montagem da serie de brilho
    brilho = []
    for ff in range(framei, framef):
        print (ff)

        # seleciona frame        
        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)

        # leitura do frame
        ret, frame = cap.read()

        # seleciona o ponto para retirar o valor de brilho
        point = frame[p]

        # funcao para o calculo do brilho
        pb = np.sum(point[0]+point[1]+point[2])/3 

        # concatena valores de bilho
        brilho.append(pb)

    # salva serie de bilho
    df = pd.DataFrame(brilho, index=time, columns=['brilho'])
    df.index.name = 'time'
    df.to_csv(pth_out + 'brilho_{}.csv'.format(fln.split('.')[0]), float_format='%.2f')
