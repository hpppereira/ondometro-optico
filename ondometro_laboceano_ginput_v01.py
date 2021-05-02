"""
Ondometro optico no laboceano com ginput
04/06/2018

CAM 1 - Central - altura - 480
CAM 3 - Central - altura - 240

video1 - camera superior
video2 - camera inferior
"""

# importa bibliotecas
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import signal

plt.close('all')

def carrega_video(pathname):
    """
    Descricao:
    - Carrega o video
    Entrada:
    - caminho e nome do arquivo de video
    Saida:
    - objeto do video
    """

    # print ('- Carregando video...')

    cap = cv2.VideoCapture(pathname)

    # if cap.isOpened():
        # print ('Video carregado! =)')
    fps = cap.get(cv2.CAP_PROP_FPS)
    # else:
        # print ('Video nao carregado! =(')
    return cap, fps

def acha_frame_inicial_e_final(timei, timef, fps):
    """
    Descricao:
    - Acha o numero do frame inicial e final do video com base no tempo
    Entrada:
    - timei: tempo inicial (formato MM:SS)
    - timef: tempo final (formato MM:SS)
    - fps: frames por segundos
    Saida:
    - nframei: numero do frame inicial
    - nframef: numero do frame final
    """

    # print ('- Achando numeros dos frames inicial e final...')

    # tempo inical e final do filme
    dtime = pd.to_datetime([timei, timef], format='%M:%S')
    # tempo em timedelta (para converter para total_seconds)
    timei = dtime[0] - pd.Timestamp('1900')
    timef = dtime[1] - pd.Timestamp('1900')
    # duracao do video em time delta
    dur = dtime[1] - dtime[0]
    # duracao do video em segundos
    durs = dur.total_seconds()
    # numero do frame inicial e final (maquina de 30 fps)
    nframei = int(timei.total_seconds() * fps)
    nframef = int(timef.total_seconds() * fps)
    # print ('Frame inicial {nframei}, Frame final {nframef}'.format(nframei=nframei, nframef=nframef))
    return nframei, nframef

def carrega_frame(cap, ff):
    """
    Descricao:
    - Carrega frame com base no numero do frame
    Entrada:
    - cap: objeto do video
    Saida:
    - frame: frame do tempo selecionado
    """

    # print ('- Carregando frame...')
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    # frame = frame[600:1000,800:]
    # print ('Frame {} carregado!! =)'.format(ff))
    return frame

def calcula_brilho(frame):
    """
    Calcula a intensidade luminosa
    Entrada:
    - frame: frame com 3 dimensoes
    Saida:
    - luminancia: com 2 dimensoes
    """

    B, G, R = frame.T
    brilho = (B + G + R) / 3
    return brilho.T

def calcula_luminancia(frame):
    """
    Calcula a luminosidade com base no RGB
    Entrada:
    - frame: frame com 3 dimensoes
    Saida:
    - luminancia: com 2 dimensoes
    """

    B, G, R = frame.T
    luminancia = 0.17697 * R + 0.81240 * G + 0.010603 * B
    return luminancia.T

def recupera_valor_pixel(frame, x, y):
    """
    Pega valor do pixel em x,y de um frame 1D
    Entrada:
    - frame: frame com 2 dimensao
    Saida:
    - valor do pixel em  x,y
    """
    valor_pixel = frame[y, x]
    return valor_pixel

def achar_horizonte(frame):
    pass

def achar_nivel_medio():
    pass

def achar_crista(frame):
    pass

def calcula_periodo_pico(serie, fps, nfft):
    """
    Calcula periodo de pico
    Entrada:
    Saida:
    """

    # calculo do espectro 1d
    f, s1 = signal.welch(serie, fs=fps, window='hann',
                         nperseg=nfft, noverlap=nfft/2, nfft=None,
                         detrend='constant', return_onesided=True,
                         scaling='density', axis=-1)

    return f, s1

def calcular_direcao_slope_array():
    pass

def plota_figura(cont_frame, nfi1, nff1, eixo_x, serie_pixel, pathout):

    gs = gridspec.GridSpec(3, 1)
    fig = plt.figure(figsize=(9, 9))
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eixo_x, serie_pixel,'b-')
    ax1.legend(['brilho','luminancia','cinza'])
    ax1.plot(cont_frame, serie_pixel[-1],'ro')
    ax1.set_xlim(nfi1, nff1)
    ax1.set_ylabel('Pixel')
    ax1.set_xlabel('Tempo (Num. Frames)')
    ax2 = fig.add_subplot(gs[1:,0])
    ax2.plot(x, y, 'ro')
    ax2.imshow(im1)
    fig.savefig('{pathout}fig_{cont_frame}'.format(pathout=pathout, cont_frame=str(cont_frame).zfill(3)))
    plt.close('all')
    return


if __name__ == '__main__':

    # dados de entrada


    # posicao para o calculo do periodo
    x, y = 1411, 792

    # paths dos videos
    video1 = os.path.join(os.environ['HOME'] + '/Documents/ondometro_videos/laboceano/CAM1/T100/', 'T100_570003_CAM1.avi')
    # video2 = os.path.join(os.environ['HOME'] + '/Documents/ondometro_videos/laboceano/CAM1/T100/', 'T100_570003_CAM3.avi')

    # path de saida das imagens
    pathout = os.environ['HOME'] + '/Documents/teste9/'

    # numero do frame inicial e final do trecho do video a processar
    timei, timef = '00:15', '01:15'

    # chama funcoes

    cap1, fps1 = carrega_video(video1)
    # cap2, fps2 = carrega_video(video2)

    nfi1, nff1 = acha_frame_inicial_e_final(timei, timef, fps1)
    # nfi2, nff2 = acha_frame_inicial_e_final(timei, timef, fps2)

    eixo_x = []
    serie_pixel_bri = []
    serie_pixel_lum = []
    serie_pixel_gray = []
    for cont_frame in np.arange(nfi1, nff1):

        eixo_x.append(cont_frame)

        print ('{cont_frame} --> {nff1}'.format(cont_frame=cont_frame, nff1=nff1))

        im1 = carrega_frame(cap1, ff=cont_frame)

        lum1 = calcula_luminancia(im1)

        bri1 = calcula_brilho(im1)

        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

        pixel_bri = recupera_valor_pixel(bri1, x, y)
        pixel_lum = recupera_valor_pixel(lum1, x, y)
        pixel_gray = recupera_valor_pixel(gray1, x, y)

        serie_pixel_bri.append(pixel_bri)
        serie_pixel_lum.append(pixel_lum)
        serie_pixel_gray.append(pixel_gray)

        plota_figura(cont_frame, nfi1, nff1, eixo_x, serie_pixel_lum, pathout)

    nfft = len(serie_pixel_lum)/2
    f, s1 = calcula_periodo_pico(serie_pixel_lum, fps1, nfft)

    # salva serie de brilho de um ponto x, y
    np.savetxt('../out/serie_pixel_lum.txt', serie_pixel_lum, fmt='%.3f')