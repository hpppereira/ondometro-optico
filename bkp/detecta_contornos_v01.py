# Importando modulos

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
################################################################################
# Criando funcoes

def carrega_video(pathname, arquivo):
    cap = cv2.VideoCapture(pathname + arquivo)
    return cap

def carrega_frames(cap, ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    return frame

def processa_imagens(frame):
    img = np.copy(frame)
    img = cv2.flip(img, 0)
    img_original = img
    img_fundo = np.copy(img_original)
    img_blur = cv2.bilateralFilter(img, 11, 15, 7)
    img_cinza = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    edges = edges = cv2.Canny(img_blur,50,100)
    return img, img_original, img_fundo, img_blur, img_cinza, edges

def aplica_gauss_blur(img_cinza):
    espraio_blur = cv2.GaussianBlur(img_cinza, (75, 75), 0)
    horizonte_blur = cv2.GaussianBlur(img_cinza, (15, 15), 0)
    return espraio_blur, horizonte_blur

def aplica_threshold(espraio_blur, horizonte_blur):
    ret, th_espraio = cv2.threshold(espraio_blur, 119, 255, cv2.THRESH_BINARY)
    ret, th_horizonte = cv2.threshold(horizonte_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, th_espraio, th_horizonte

def cria_vetor(th_espraio, th_horizonte, c):
    vetor_espraio = th_espraio[:,c]
    vetor_horizonte = th_horizonte[:,c]
    return vetor_espraio, vetor_horizonte

def aplica_derivada(vetor_espraio, vetor_horizonte):
    derivada_vetor_espraio = np.diff(vetor_espraio)
    derivada_vetor_espraio = np.concatenate(([0],derivada_vetor_espraio))
    derivada_vetor_horizonte = np.diff(vetor_horizonte)
    derivada_vetor_horizonte = np.concatenate(([0],vetor_horizonte))
    return derivada_vetor_espraio, derivada_vetor_horizonte

def acha_valores_y(derivada_vetor_espraio, deriivada_vetor_horizonte):
    try:
        id_y_espraio = np.where(derivada_vetor_espraio!=0)[0][-1]  # camera deve estar estavel
    except IndexError:
        id_y_espraio = img.shape[0]
    try:
        id_y_horizonte = np.where(derivada_vetor_horizonte!=0)[0][-1]
    except IndexError:
        id_y_horizonte = horizonte[int('%d'% (c-1))]
    return id_y_espraio, id_y_horizonte

def cria_listas_vazias():
    espraio = []
    horizonte = []
    return espraio, horizonte

def preenche_listas(id_y_espraio, id_y_horizonte):
    espraio.append(id_y_espraio)
    horizonte.append(id_y_horizonte)

def acha_cristas(edges, img_fundo):
    edges[:max(horizonte) + 20] = 0
    edges[(min(espraio) - 20):720] = 0
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        cv2.drawContours(img_fundo, contour, -1, (0, 255, 0), 3)
    return contours, img_fundo

def ilustra_resultado(edges, img_fundo):
    plt.figure()
    plt.subplot(121),plt.imshow(edges,cmap = 'gray')
    plt.title('Canny edge detection'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_fundo,cmap = 'gray'),plt.plot(espraio, color='red'),plt.plot(horizonte, color='yellow')
    plt.title('Detecção de contornos'), plt.xticks([]), plt.yticks([])

    plt.show()


################################################################################
# Decidindo qual partes do programa rodar

if __name__ == '__main__':

    processar_frames = True
    detectar_contornos = True
    visualizar_resultados = True
    salvar_imagens = True

################################################################################
# Dados de entrada

    pathname = os.environ['HOME'] + '/Documentos/Lioc/Ondometro/ondometro_videos/praia_seca_20180405/Cel_Victor/filme_02/'
    path_to_save = '/Documentos/Lioc/Ondometro/ondometro_frames/'
    arquivo = '20180405_165833.mp4'

    inicio = 100 ## inicio do intervalo a ser processado (em segundos)
    fim = 110 ## fim do intervalo a ser processado
    frame_inicial = inicio * 30
    frame_final = fim * 30
    intervalo = 50 ## intervalo entre os frames a serem processados (em frames)

################################################################################
# Deteccao de contornos

    if processar_frames == True:

        cap = carrega_video(pathname, arquivo)

        for ff in np.arange(frame_inicial, frame_final, intervalo):

            frame = carrega_frames(cap, ff)
            img, img_original, img_fundo, img_blur, img_cinza, edges = processa_imagens(frame)
            espraio_blur, horizonte_blur = aplica_gauss_blur(img_cinza)
            ret, th_espraio, th_horizonte =  aplica_threshold(espraio_blur, horizonte_blur)

            espraio, horizonte = cria_listas_vazias()

            if detectar_contornos == True:

                for c in np.arange(0,img.shape[1]):

                    vetor_espraio, vetor_horizonte = cria_vetor(th_espraio, th_horizonte, c)
                    derivada_vetor_espraio, derivada_vetor_horizonte = aplica_derivada(vetor_espraio, vetor_horizonte)
                    id_y_espraio, id_y_horizonte = acha_valores_y(derivada_vetor_espraio, derivada_vetor_horizonte)

                    if c == 0:
                        preenche_listas(id_y_espraio, id_y_horizonte)
                    else:
                        if espraio[c-1] - 6 < id_y_espraio < espraio[c-1] + 6:
                            preenche_listas(id_y_espraio, id_y_horizonte)
                        else:
                            id_y_espraio = espraio[c-1]
                            preenche_listas(id_y_espraio, id_y_horizonte)

                contours, img_fundo = acha_cristas(edges, img_fundo)

                if visualizar_resultados == True:

                    ilustra_resultado(edges, img_fundo)

                    if salvar_imagens == True:

                        nome_das_imagens = arquivo[:15] + '_frame%d'%ff
                        plt.savefig(os.environ['HOME'] + path_to_save + nome_das_imagens)
