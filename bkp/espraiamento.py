"""
Programa para deteccao do nivel de espraiaamento e esmaramento

Procedimentos:
- tetectar a primmeira derivada de baixo para cima

Referencias:
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
https://docs.opencv.org/3.3.0/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html
https://docs.opencv.org/trunk/db/d8e/tutorial_threshold.html
"""

#Importar bibliotecas

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
################################################################################

cv2.destroyAllWindows()
plt.close('all')


def carrega_video(pathname, nome_video):
    cap = cv2.VideoCapture(pathname + nome_video)
    return cap

def calcula_dimensoes(cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #largura
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #altura
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #numero de frames
    return width, height, length

def carrega_frames(cap, ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    return frame

def gera_imagem(frame):
    img = np.copy(frame)
    img = cv2.flip(img, 0)
    img_original = img
    return img, img_original

def converte_para_cinza(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def aplica_gauss_blur(img):
    espraio_blur = cv2.GaussianBlur(img, (75, 75), 0)
    horizonte_blur = cv2.GaussianBlur(img, (15, 15), 0)
    cristas_blur = cv2.GaussianBlur(img, (15, 15), 0)
    return espraio_blur, horizonte_blur, cristas_blur

def aplica_threshold(espraio_blur, horizonte_blur, cristas_blur):
    ret, th_espraio = cv2.threshold(espraio_blur, 119, 255, cv2.THRESH_BINARY)
    ret, th_horizonte = cv2.threshold(horizonte_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, th_cristas = cv2.threshold(cristas_blur, 130, 255, cv2.THRESH_BINARY)
    return ret, th_espraio, th_horizonte, th_cristas

def cria_vetor(th_espraio, th_horizonte, c):
    vetor_espraio = th_espraio[:,c] # vetor para detectar espraio
    vetor_horizonte = th_horizonte[:,c] # vetor para detectar horizonte
    return vetor_espraio, vetor_horizonte

def aplica_derivada(vetor_espraio, vetor_horizonte):
    derivada_vetor_espraio = np.diff(vetor_espraio)
    derivada_vetor_espraio = np.concatenate(([0],derivada_vetor_espraio))
    derivada_vetor_horizonte = np.diff(vetor_horizonte)
    derivada_vetor_horizonte = np.concatenate(([0],vetor_horizonte))
    return derivada_vetor_espraio, derivada_vetor_horizonte

def acha_valores_y(derivada_vetor_espraio, deriivada_vetor_horizonte):
    id_y_espraio = np.where(derivada_vetor_espraio!=0)[0][-1]  # camera deve estar estavel
    id_y_horizonte = np.where(derivada_vetor_horizonte!=0)[0][-1]
    return id_y_espraio, id_y_horizonte

def aplica_para_cristas(th_cristas, id_y_horizonte, id_y_espraio, c):
    vetor_cristas = th_cristas[id_y_horizonte:id_y_espraio, c] # vetor para detectar horizonte
    derivada_vetor_cristas = np.diff(vetor_cristas)
    derivada_vetor_cristas = np.concatenate(([0],derivada_vetor_cristas))
    id_y_cristas = np.where(derivada_vetor_cristas==255)[0]
    return vetor_cristas, derivada_vetor_cristas, id_y_cristas

def cria_listas_vazias():
    espraio = []
    horizonte = []
    cristas = []
    return espraio, horizonte, cristas

def preenche_listas(id_y_espraio, id_y_horizonte, id_y_cristas):
    espraio.append(id_y_espraio)
    horizonte.append(id_y_horizonte)
    cristas.append(id_y_cristas + id_y_horizonte)

def plota_no_frame(espraio, horizonte, cristas):
    ww = 30
    espraio = espraio.rolling(window=ww, center=False).mean()
    espraio.iloc[:ww,0] = espraio.iloc[ww,0]
    plt.figure()
    plt.imshow(img_original)
    plt.plot(range(img.shape[1]), espraio, 'k')
    plt.plot(range(img.shape[1]), horizonte, 'r')
    plt.plot(range(img.shape[1]), cristas, 'y')

def calcula_espraio_medio(lista_espraios, espraio):
    lista_espraios.append(espraio)
    espraio_medio_linha = np.mean(lista_espraios)
    return espraio_medio_linha

if __name__ == '__main__':

    acha_xy = True
    plota_na_imagem = False
    salva_imagens = False

    # Dados de entrada:
    pathname = os.environ['HOME'] + \
               '/Documentos/Lioc/Ondometro/ondometro_videos/praia_seca_20180405/Cel_Victor/filme_02/'

    nome_video = '20180405_165833.mp4'

    if acha_xy == True:

        cap = carrega_video(pathname, nome_video)
        width, height, length = calcula_dimensoes(cap)
        conjunto_espraios = pd.DataFrame(index=np.arange(0,1280))
        lista_espraios = []
        #espraio_medio_pontual = []

        for ff in np.arange(2000, 9000, 100):

            frame = carrega_frames(cap, ff)
            img, img_original = gera_imagem(frame)
            img = converte_para_cinza(img)
            espraio_blur, horizonte_blur, cristas_blur = aplica_gauss_blur(img)
            ret, th_espraio, th_horizonte, th_cristas =  aplica_threshold(espraio_blur, horizonte_blur, cristas_blur)

            espraio, horizonte, cristas = cria_listas_vazias()

            for c in np.arange(0,img.shape[1]):

                vetor_espraio, vetor_horizonte = cria_vetor(th_espraio, th_horizonte, c)
                derivada_vetor_espraio, derivada_vetor_horizonte = aplica_derivada(vetor_espraio, vetor_horizonte)
                id_y_espraio, id_y_horizonte = acha_valores_y(derivada_vetor_espraio, derivada_vetor_horizonte)
                vetor_cristas, derivada_vetor_cristas, id_y_cristas = aplica_para_cristas(th_cristas, id_y_horizonte, id_y_espraio, c)

                if len(id_y_cristas) > 0:
                    id_y_cristas = id_y_cristas[0]
                else:
                    id_y_cristas = np.nan

                preenche_listas(id_y_espraio, id_y_horizonte, id_y_cristas)

            espraio_medio_linha = calcula_espraio_medio(lista_espraios, espraio)

            espraio = pd.DataFrame(espraio)
            conjunto_espraios = conjunto_espraios.join(espraio, how = 'right', rsuffix='%s'%ff)
            matriz_espraios = conjunto_espraios.as_matrix()

            if plota_na_imagem == True:

                plota_no_frame(espraio, horizonte, cristas)

                plt.show()

            if salva_imagens == True:

                plt.savefig('/home/joao2many/Documentos/Lioc/Ondometro/ondometro_frames/praia_seca_20180405/Cel_Nelson/filme_02/frame%d'%ff)

        espraio_medio_pontual = []

        for ll in np.arange(0,1280):
            espraio_medio_pontual.append(np.mean(matriz_espraios[ll,:]))

        espraio_medio_pontual = pd.DataFrame(espraio_medio_pontual)

        plt.figure('espraio médio')
        plt.imshow(img_original)
        plt.axhline(y=espraio_medio_linha, color= 'cyan')
        plt.plot(range(img.shape[1]), espraio_medio_pontual,'b')
        plt.show()
