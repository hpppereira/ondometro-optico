'''Modulos importados'''

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd

################################################################################
cv2.destroyAllWindows()
plt.close('all')
################################################################################

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
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq_img = cv2.equalizeHist(img_cinza)
    return img, img_original, img_cinza, eq_img

def prepara_horizonte(img_cinza):
    mediana_horizonte = cv2.medianBlur(img_cinza,51)
    contorno_horizonte = cv2.Canny(mediana_horizonte, 10, 80)
    horizonte = []
    return mediana_horizonte, contorno_horizonte, horizonte

def prepara_espraio(img_cinza, eq_img, metodo_espraio):
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/2, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    espraio = []


    #kernel = np.ones((5,5), np.float32) / 25
    #filtro2d = cv2.filter2D(eq_img, -1, kernel)
    #contorno_espraio = cv2.Canny(filtro2d, 75, 120)
    #kernel = np.ones((9,9),np.uint8)
    #contorno_espraio = cv2.morphologyEx(contorno_espraio, cv2.MORPH_CLOSE, kernel)
    #ret, contorno_espraio = cv2.threshold(eq_img, 110, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((5,5),np.uint8)
    #contorno_espraio = cv2.erode(contorno_espraio, kernel,iterations=1)
    #contorno_espraio = cv2.filter2D(contorno_espraio, cv2.CV_8UC3, g_kernel)
    #espraio = []
    if metodo_espraio == 1:
        #------------------> metodo 1
        ret, contorno_espraio = cv2.threshold(eq_img, 115, 255, cv2.THRESH_BINARY)

    if metodo_espraio == 2:
        #------------------> metodo 2
        eq_img = cv2.fastNlMeansDenoising(eq_img,None,51,7,21)
        eq_img = cv2.medianBlur(eq_img, 9)
        ret, contorno_espraio = cv2.threshold(eq_img, 110, 255, cv2.THRESH_BINARY)
        #kernel = np.ones((99,99),np.uint8) ##????
        #contorno_espraio = cv2.morphologyEx(contorno_espraio, cv2.MORPH_CLOSE, kernel) ##????
        #kernel = np.ones((5,5),np.uint8) ##????
        #contorno_espraio = cv2.dilate(contorno_espraio, kernel, iterations=1) ##????
    if metodo_espraio == 3:
        #------------------> metodo 3
        eq_img = cv2.fastNlMeansDenoising(eq_img,None,51,7,21)
        filtered_img = cv2.filter2D(eq_img, cv2.CV_8UC3, g_kernel)
        ret, contorno_espraio = cv2.threshold(filtered_img, 200, 255, cv2.THRESH_BINARY)

    return contorno_espraio, espraio,

def detecta_horizonte(contorno_horizonte, horizonte):
    colunas_horizonte = contorno_horizonte[:,c]
    derivada_horizonte =  np.diff(colunas_horizonte)
    id_y_horizonte = np.where(derivada_horizonte!=0)[-1][0]
    horizonte.append(id_y_horizonte)
    return derivada_horizonte, id_y_horizonte, horizonte

def detecta_espraio(contorno_espraio, espraio):
    colunas_espraio = contorno_espraio[:,c]
    derivada_espraio = np.diff(colunas_espraio)
    try:
        id_y_espraio = np.where(derivada_espraio!=0)[0][-1]
    except IndexError:
        id_y_espraio = np.NaN

    if c == 0:
        espraio.append(id_y_espraio)
    else:
        if espraio[c-1] - 20 < id_y_espraio < espraio[c-1] + 20:
            espraio.append(id_y_espraio)
        else:
            id_y_espraio = espraio[c-1]
            espraio.append(id_y_espraio)
    #espraio.append(id_y_espraio)
    return derivada_espraio, id_y_espraio, espraio

def pas(df, i, output, alpha=0.01, input_column='input'):
    input_i = df.loc[i][input_column]
    output_i_1 = df.loc[i-1][output] if i > 0 else df.loc[0][input_column]
    output_i = output_i_1 + alpha * (input_i - output_i_1)
    return output_i

def alisa_horizonte(horizonte):
    a, b= np.polyfit(np.arange(len(horizonte)),horizonte,1)
    y = a * np.arange(len(horizonte))+b
    return y

def serie_temporal_contorno(contorno):
    serie_temporal = []
    media_contorno = np.mean(contorno)
    serie_temporal.append(media_contorno)
    return serie_temporal

def plota_contornos(horizonte):

    plt.figure('frame %d' %ff)
    plt.imshow(img_original)
    plt.plot(horizonte, 'r')
    plt.plot(espraio, 'b')
    plt.plot(df.two_pole_low_pass,'orange')
    plt.show()

def cria_mascara(eq_img, df, horizonte):
    mascara = np.zeros(eq_img.shape[:2],dtype="uint8")

    for c in np.arange(0, eq_img.shape[1]):
        mascara[int(horizonte[c]):int(df.two_pole_low_pass[c]), c] = 255
    mascara = cv2.bitwise_and(eq_img, eq_img, mask=mascara)
    return mascara

# Compute the frame difference
def frame_diff(prev_frame, cur_frame, next_frame):
    # Absolute difference between current frame and next frame
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)

    # Absolute difference between current frame and # previous frame
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)

    # Return the result of bitwise 'AND' between the # above two resultant images
    return...
################################################################################

if __name__ == '__main__':

    visualizar_contornos = True
    metodo_espraio = 2

    # Dados de entrada:
    pathname = os.environ['HOME'] + \
               '/Documentos/Lioc/Ondometro/ondometro_videos/praia_seca_20180405/Cel_Victor/filme_02/'

    nome_video = '20180405_165833.mp4'


    cap = carrega_video(pathname, nome_video)
    width, height, length = calcula_dimensoes(cap)

    for ff in np.arange(4400, 6000, 2000):

        print('----> Frame %d' %ff)

        frame = carrega_frames(cap, ff)
        img, img_original, img_cinza, eq_img = gera_imagem(frame)
        mediana_horizonte, contorno_horizonte, horizonte = prepara_horizonte(img_cinza)
        contorno_espraio, espraio = prepara_espraio(img_cinza, eq_img, metodo_espraio)

        for c in np.arange(0, img_original.shape[1]):

            derivada_horizonte, id_y_horizonte, horizonte = detecta_horizonte(contorno_horizonte, horizonte)
            derivada_espraio, id_y_espraio, espraio = detecta_espraio(contorno_espraio, espraio)

        horizonte = alisa_horizonte(horizonte)

        data = list(map(lambda v: [espraio[v], None, None], range(len(espraio))))
        df = pd.DataFrame(data, columns=['input', 'single_pole_low_pass', 'two_pole_low_pass'])


        for i in df.index:
            output = pas(df, i, 'single_pole_low_pass');
            df.loc[i, 'single_pole_low_pass'] = output
            output = pas(df, i, 'two_pole_low_pass', alpha=0.02, input_column='single_pole_low_pass');
            df.loc[i, 'two_pole_low_pass'] = output

        serie_temporal_horizonte = serie_temporal_contorno(horizonte)
        serie_temporal_espraio = serie_temporal_contorno(df.two_pole_low_pass)

        mascara = cria_mascara(eq_img, df, horizonte)

        #df.plot()
        #plt.xlabel('time')

        if visualizar_contornos == True:

            plota_contornos(horizonte)

subtracao_frames = True

if subtracao_frames == True:


    for ff in np.arange(4400, 4401):

        cap.set(cv2.CAP_PROP_POS_FRAMES, ff-3)
        ret, frame_pre = cap.read()
        frame_pre = cv2.flip(frame_pre, 0)
        frame_pre = cv2.cvtColor(frame_pre, cv2.COLOR_BGR2GRAY)

        cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
        ret, frame_cur = cap.read()
        frame_cur = cv2.flip(frame_cur, 0)
        frame_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)

        diff_frames1 = cv2.absdiff(frame_cur, frame_pre)

        #ret, contorno_espraio = cv2.threshold(diff_frames1, 7, 255, cv2.THRESH_BINARY)

        #lista = []

        #for ll in np.arange(0,diff_frames1.shape[0]):
        #    intensidade_linha = sum(diff_frames1[ll,:])
        #    lista.append(intensidade_linha)

        lista=[]
        for col in np.arange(0,diff_frames1.shape[1]):

            col_espraio = diff_frames1[:,col]
            col_espraio = np.array((col_espraio),np.int)
            derivada = np.diff(col_espraio)

            max_coluna = pd.Series(derivada).idxmax()
            lista.append(max_coluna)

        plt.figure('motion')
        plt.imshow(diff_frames1,cmap='gray')
        plt.show()
