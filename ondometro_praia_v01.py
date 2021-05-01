'''Modulos importados'''

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import peakutils
from scipy import stats

################################################################################
cv2.destroyAllWindows()
plt.close('all')
################################################################################
'''Funcoes do programa'''

def carrega_video(caminho, nome_video):
    cap = cv2.VideoCapture(caminho + nome_video)
    return cap

def calcula_dimensoes(cap):
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #largura
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #altura
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #numero de frames
    return largura, altura, nframes

def carrega_frames(cap, ff):
    cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
    ret, frame = cap.read()
    return frame

def gera_imagem(frame):
    img = np.copy(frame)
    img_original = img
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq_img = cv2.equalizeHist(img_cinza)
    return img, img_original, img_cinza, eq_img

def acentua_bordas(img_cinza, eq_img):
    blur_img_cinza = cv2.medianBlur(img_cinza,5)
    blur_eq_img = cv2.medianBlur(eq_img,5)
    return blur_img_cinza, blur_eq_img

#def detecta_horizonte(blur_img_cinza, r):
#    kernel = np.ones((3,3),np.uint8)
#    roi = blur_img_cinza[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
#    roi = cv2.morphologyEx(roi, cv2.MORPH_ERODE, kernel)
#    canny = cv2.Canny(roi, 5,30, apertureSize=3)
#    linhas = cv2.HoughLines(canny,1,np.pi/180,200)
#    b = linhas[0][0][0]
#    a = -np.cos(linhas[0][0][1])
#    horizonte = a * np.arange(1280) + (r[1] + b)
#    return canny, linhas, horizonte

def subtrai_frames(pre_frame, img_cinza):
    pre_img_cinza = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(img_cinza, pre_img_cinza)
    return pre_img_cinza, frame_diff

def monta_histograma(altura, frame_diff):
    hist_mov = []
    for ll in np.arange(0, altura):
        soma_linha = np.sum(frame_diff[ll,:])
        hist_mov.append(soma_linha)
    hist_mov = pd.DataFrame(hist_mov)
    hist_mov = hist_mov.rolling(window=41, center=True).mean()
    hist_mov = hist_mov[0].values.tolist()
    for ll in np.arange(0, altura):
        if np.isnan(hist_mov[ll]) == True:
            hist_mov[ll] = 0
    return hist_mov

def detecta_espraio(altura, hist_mov):  ###off ---> alternativa
    pontos = []
    for pp in np.arange(0, altura):
        if 30 <= pp <= (altura - 30):
            alpha = np.var(hist_mov[pp-30:pp])/1000
            beta = np.var(hist_mov[pp:pp+30])/1000
            controle = alpha - beta
            pontos.append(controle)
        else:
            controle = 0
            pontos.append(controle)

    pontos = pd.DataFrame(pontos)
    espraio = peakutils.indexes(pontos[0], thres=0.57, min_dist=100)
    espraio = np.max(espraio)
    return pontos, espraio

def normaliza_pontos_criticos(pontos):
    p_criticos = []
    for valor in pontos:
        norm = valor/1000
        p_criticos.append(norm)
        return p_criticos

def calcula_media_movel(a, n=9) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def calcula_derivada_movel(lista, i, a=0):
    n = i + 1
    lista_diff  = np.diff(lista)
    derivada_i = np.sum(lista_diff[a:n])
    derivada_movel = []
    derivada_movel.append(derivada_i)

    while n < len(lista_diff):
        derivada_n = derivada_i + lista_diff[n] - lista_diff[a]
        derivada_i = derivada_n
        derivada_movel.append(derivada_i)
        a = a + 1
        n = n + 1
    return derivada_movel

def calcula_dispersao(lista, i):
    n = i + 1
    somatorio = np.sum(lista[:n])
    somatorio_quadrados = np.sum([y ** 2 for y in lista[:n]])
    Ai = somatorio_quadrados
    Bi = somatorio
    sigma2 = (Ai/n) - (Bi/n)**2
    return Ai, Bi, np.sqrt(sigma2)

def padroniza_listas(coluna, media_movel, derivada_movel):
    padrao = np.zeros(int((len(coluna) - len(media_movel)) / 2)) + np.NaN
    correcao = (np.zeros(int((len(coluna) - len(derivada_movel)) - (len(coluna) - len(media_movel)))) + np.NaN)
    derivada_movel = (correcao.tolist() + padrao.tolist()) + derivada_movel + padrao.tolist()
    media_movel = padrao.tolist() + media_movel + padrao.tolist()
    return media_movel, derivada_movel

def pontos_notaveis(largura, imagem, a, b, f): #a=-1/b=0: de cima para baixo; a=0/b=-1: de baixo para cima, f -->fator dispersao
    pontos_notaveis = []
    for cc in np.arange(0, largura):
        coluna = imagem[:,cc].tolist()
        media_movel = calcula_media_movel(coluna, n=9)
        media_movel = media_movel.tolist()
        derivada_movel = calcula_derivada_movel(media_movel, i=4,a=0)
        disp_movel = []
        Ai, Bi, disp_i = calcula_dispersao(coluna, i=0)
        disp_movel.append(disp_i)
        i = 1
        while i < altura:
            n = i + 1
            disp_n = np.sqrt(((Ai + (coluna[i])**2)/n) - ((Bi + coluna[i])/n)**2)
            disp_movel.append(disp_n)
            Ai = Ai + (coluna[i])**2
            Bi = Bi + coluna[i]
            i = i + 1

        media_movel, derivada_movel = padroniza_listas(coluna, media_movel, derivada_movel)
        d = {'media_movel':media_movel,'derivada_movel':derivada_movel,'disp_movel':disp_movel}
        df = pd.DataFrame(data=d,index=np.arange(altura))
        try:
            pontos_notaveis.append(np.where(df.derivada_movel < (-f * df.disp_movel))[a][b])
        except IndexError:
            pontos_notaveis.append(derivada_movel.index(np.nanmin(derivada_movel)))
    return df, pontos_notaveis

def pontos_cristas(largura, imagem, horizonte, espraio, f):
    p_crista1 = []
    p_crista2 = []
    p_crista3 = []
    for cc in np.arange(0, largura):
        coluna = imagem[:,cc].tolist()
        lista = []
        disp_movel = []
        Ai, Bi, disp_i = calcula_dispersao(coluna, i=0)
        disp_movel.append(disp_i)
        i = 1
        while i < altura:
            n = i + 1
            disp_n = np.sqrt(((Ai + (coluna[i])**2)/n) - ((Bi + coluna[i])/n)**2)
            disp_movel.append(disp_n)
            Ai = Ai + (coluna[i])**2
            Bi = Bi + coluna[i]
            i = i + 1
        for i in np.arange(altura):
            if 4 < i < (altura - 4):
                alpha = np.mean(coluna[i-4:i])
                beta = np.mean(coluna[i:i+4])
                perfil = beta - alpha
                lista.append(perfil)
            else:
                lista.append(0)
        d = {'lista':lista,'disp_movel':disp_movel}
        df = pd.DataFrame(data=d,index=np.arange(altura))
        try:
            p_crista1.append(int(np.max(horizonte)) + np.where(df.lista[int(np.max(horizonte)):int(np.min(espraio))] < (-f * df.disp_movel[int(np.max(horizonte)):int(np.min(espraio))]))[-1][0])
        except IndexError:
            p_crista1.append(np.nan)
        try:
            p_crista2.append(int(np.max(horizonte)) + np.where(df.lista[int(np.max(horizonte)):int(np.min(espraio))] < (-f * df.disp_movel[int(np.max(horizonte)):int(np.min(espraio))]))[-1][1])
        except IndexError:
            p_crista2.append(np.nan)
        try:
            p_crista3.append(int(np.max(horizonte)) + np.where(df.lista[int(np.max(horizonte)):int(np.min(espraio))] < (-f * df.disp_movel[int(np.max(horizonte)):int(np.min(espraio))]))[-1][2])
        except IndexError:
            p_crista3.append(np.nan)
    return  df, p_crista1, p_crista2, p_crista3


#def sigma_reta(pontos_notaveis, y):
#    dist_quads = []
#    for i in np.arange(len(pontos_notaveis)):
#        dist_quad = (pontos_notaveis[i] - y[i])**2
#        dist_quads.append(dist_quad)
#    sigma = np.sqrt(np.sum(dist_quads)/len(pontos_notaveis))
#    return dist_quads, sigma

def detecta_feicao(pontos_notaveis, largura):
    moda = stats.mode(pontos_notaveis)[0][0]
    y = []
    x = []
    for i in np.arange(len(pontos_notaveis)):
        if moda - 20 < pontos_notaveis[i] < moda + 20:
            y.append(pontos_notaveis[i])
            x.append(i)
    a, b = np.polyfit(x, y, 1)
    feicao = a * np.arange(largura) + b
    return feicao

#def elimina_pontos_distantes(dist_quads, sigma):
#    x = np.arange(len(pontos_horizonte))
#    y = pontos_horizonte
#    Xc = x
#    Yc = np.array(y)
#    a, b = np.polyfit(Xc, Yc, 1)
#    Xr = x
#    Yr = a * Xr + b
#    Dq = (Yc - Yr)**2
#    sigma = np.sqrt(np.sum(Dq)/len(Yc))
#    controleYc = []
#    controleXc = []
#    controleA = []
#    controleB = []
#    while sigma > 3:
#        Yc = np.delete(Yc, np.argmax(Dq))
#        Xc = np.delete(Xc, np.argmax(Dq))
#        Xr = np.delete(Xr, np.argmax(Dq))
#        a, b = np.polyfit(Xc, Yc, 1)
#        Yr = a * Xr + b
#        Dq = (Yc - Yr)**2
#        sigma = np.sqrt(np.sum(Dq)/len(Yc))
#    return df, pontos_horizonte
################################################################################
'''dados de entrada'''


caminho = os.environ['HOME'] + '/Documentos/Lioc/projeto/videos/Leme_20181010/Dist_cam_1m/Cam_inf/'

nome_video = '20181010_v01_1m_inf.mp4'

################################################################################
'''programa'''

cap = carrega_video(caminho, nome_video)
largura, altura, nframes = calcula_dimensoes(cap)

###############----> Selecionar a regiao de interesse ##########################
frame = carrega_frames(cap, 2000)
img, img_original, img_cinza, eq_img = gera_imagem(frame)
cv2.destroyAllWindows()
################################################################################

for ff in np.arange(2000, 2001):
    frame = carrega_frames(cap,ff)
    img, img_original, img_cinza, eq_img = gera_imagem(frame)

    df_horizonte, pontos_horizonte = pontos_notaveis(largura, img_cinza, a=-1, b=0, f=1)
    horizonte = detecta_feicao(pontos_horizonte, largura)

    pre_frame = carrega_frames(cap, ff-3)
    pre_img_cinza, frame_diff = subtrai_frames(pre_frame, img_cinza)
    hist_mov = monta_histograma(altura, frame_diff)
    #pontos, espraio = detecta_espraio(altura, hist_mov)
    df_espraio, pontos_espraio = pontos_notaveis(largura, frame_diff, a=0, b=-1, f=0.5)
    espraio = detecta_feicao(pontos_espraio, largura)

    df_cristas, p_crista1, p_crista2, p_crista3 = pontos_cristas(largura, img_cinza, horizonte, espraio, f=1.5)

    plt.figure('frame %d' %ff)
    plt.imshow(img_cinza, cmap='gray')
    plt.plot(horizonte, c='red')
    plt.plot(espraio, c='cyan')
    plt.plot(p_crista1,'.', markersize=1, c='yellow')
    plt.plot(p_crista1,'.', markersize=1, c='lime')
    plt.plot(p_crista1,'.', markersize=1, c='pink')

plt.show()
