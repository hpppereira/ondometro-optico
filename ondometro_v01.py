"""
Programa principal do ondometro optico
04/06/2018

CAM 1 - Central - altura - 480
CAM 3 - Central - altura - 240
"""

# importa bibliotecas
import os
import numpy as np
import cv2
import pandas as pd

def carrega_video(pathname):
    """
    Descricao:
    - Carrega o video
    Entrada:
    - caminho e nome do arquivo de video
    Saida:
    - objeto do video
    """
    print ('- Carregando video...')
    cap = cv2.VideoCapture(pathname)

    if cap.isOpened():
        print ('Video carregado! =)')
    else:
        print ('Video nao carregado! =(')
    return cap

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

    print ('- Achando numeros dos frames inicial e final...')

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
    print ('Frame inicial {nframei}, Frame final {nframef}'.format(nframei=nframei, nframef=nframef))
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

    print ('- Carregando frame...')
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    # frame = frame[600:1000,800:]
    print ('Frame {} carregado!! =)'.format(ff))
    return frame

def sincroniza_imagens():
    """
    ???
    """
    pass

def achar_horizonte(frame):
    """
    Descricao:
    - Achar linha do horizonte
    Entrada:
    -
    Saida:
    -
    """

    ll, cc = frame.shape
    # for c in range(cc):
    col = frame[:,0].astype(float)
    der1 = np.diff(col)
    der1 = np.concatenate((der1,[der1[-1]]))
    return col, der1


def achar_nivel_medio():
    """
    Descricao:
    - Achar linha nivel medio
    Entrada:
    -
    Saida:
    -
    """
    pass

def achar_crista(frame):
    """
    Descricao:
    - Achar crista
    Entrada:
    -
    Saida:
    -
    """

if __name__ == '__main__':

    # dados de entrada
    video = 'T100_570003_CAM1.avi'
    pathname = os.path.join(os.environ['HOME'] + '/Documents/ondometro_videos/laboceano/CAM1/T100/', video)
    timei, timef = '00:15', '00:25'
    fps = 30

    # chama funcoes
    cap = carrega_video(pathname)
    nframei, nframef = acha_frame_inicial_e_final(timei, timef, fps)
    frame = carrega_frame(cap, ff=nframei)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    col, der1 = achar_horizonte(frame)
