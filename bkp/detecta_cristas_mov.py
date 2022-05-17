import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import peakutils

################################################################################
cv2.destroyAllWindows()
plt.close('all')
################################################################################


def carrega_video(pathname, nome_video):
    cap = cv2.VideoCapture(pathname + nome_video)
    return cap

def calcula_dimensoes(cap):
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #largura
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #altura
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #numero de frames
    return largura, altura, n_frames

def carrega_frames(cap, ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    return frame

def gera_imagem(frame):
    img = np.copy(frame)
    img = cv2.flip(img, 0)
    img_original = img
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq_img = cv2.equalizeHist(img_cinza)
    return img, img_original, img_cinza, eq_img

def subtrai_frames(img_pos_cinza, img_cinza):
    mov_img = cv2.absdiff(img_pos_cinza, img_cinza)
    return mov_img

def somatorio_linhas(mov_img):
    hist = []
    for linha in np.arange(0, altura):
        somatorio_linha = sum(mov_img[linha,:])
        hist.append(somatorio_linha)
    hist = pd.DataFrame(hist)
    return hist

################################################################################

if __name__ == '__main__':


    ##########################################################
    # Dados de entrada:

    pathname = os.environ['HOME'] + \
               '/Documentos/Lioc/Ondometro/ondometro_videos/praia_seca_20180405/Cel_Victor/filme_02/'

    nome_video = '20180405_165833.mp4'

    inicio =2630
    final= 2700
    intervalo = 100
    ##########################################################


    cap = carrega_video(pathname, nome_video)
    largura, altura, n_frames = calcula_dimensoes(cap)

    for ff in np.arange(inicio, final, intervalo):

        frame_pos = carrega_frames(cap, ff - 3)
        frame_atual = carrega_frames(cap, ff)
        img_pos, img_pos_original, img_pos_cinza, eq_img_pos = gera_imagem(frame_pos)
        img, img_original, img_cinza, eq_img = gera_imagem(frame_atual)
        mov_img = subtrai_frames(img_pos_cinza, img_cinza)

        hist = somatorio_linhas(mov_img)
        hist = hist.rolling(window=21, center=True).mean()

        indices = peakutils.indexes(hist[0], thres=0.5, min_dist=100)

        plt.figure('checagem dos picos, frame%d' %ff)
        plt.plot(hist[0]/10)
        plt.plot(hist[0][indices]/10,'.', c='red')

        ###############################################
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(img_original, cmap='gray')
        ax1.set_title('Imagem original')
        ax2.imshow(mov_img, cmap='gray')
        ax2.set_title('Subtração de frames anteriores')
        plt.figure('frames %d e %d'%(ff, ff-3))
        plt.imshow(mov_img, cmap='gray')
        plt.title('histograma do somatório dos pixels em cada linha')
        plt.plot(hist/10, np.arange(0,720), color='#00ff3bff', ls=':')
        plt.show()
        ###############################################

        indices_crista = []
        indices_espraio = []
        teta = 0
        valores_espraio = []
        valores_crista = []

        while teta <= 15:
            matriz_rotacao = cv2.getRotationMatrix2D((largura/2,altura/2),teta,1)
            rot_mov_img = cv2.warpAffine(mov_img,matriz_rotacao,(largura,altura))
            rot_hist = somatorio_linhas(rot_mov_img)
            rot_hist = rot_hist.rolling(window=21, center=True).mean()
            rot_indices = peakutils.indexes(rot_hist[0], thres=0.5, min_dist=100)
            indices_espraio.append(max(rot_indices))
            indices_crista.append(min(rot_indices))
            valores_espraio.append(rot_hist[0][max(rot_indices)])
            valores_crista.append(rot_hist[0][min(rot_indices)])
            teta = teta + 1

        teta_espraio = np.argmax(valores_espraio)
        teta_crista = np.argmax(valores_crista)
        y_rot_espraio = indices_espraio[np.argmax(valores_espraio)]
        y_rot_crista = indices_crista[np.argmax(valores_crista)]
        a_espraio = np.tan(teta_espraio)
        a_crista = np.tan(teta_crista)
        b_espraio = y_rot_espraio / np.cos(teta_espraio)
        b_crista = y_rot_crista / np.cos(teta_crista)

        reta_espraio = (a_espraio * np.arange(0, largura)) + b_espraio
        reta_crista = (a_crista * np.arange(0, largura)) + b_crista

        plt.figure('frame %d' %ff)
        plt.imshow(img)
        plt.plot(reta_crista, c='orange')
        plt.plot(reta_espraio, c='red')
        plt.show()

        #plt.figure('checagem dos picos de intensidade')
        #plt.plot(hist)
        #plt.plot(hist[0][indexes],'.', color='#ff2a51')
        #plt.show()
