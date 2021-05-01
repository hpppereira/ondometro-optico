"""
Optical Wave Measurement (OWM)

- read video
- read selected frames
- identify creasts
- ...

"""

# import libraries
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import signal
import peakutils

plt.close('all')

def carregar_video(pathname):
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

def achar_frame_inicial_e_final(timei, timef, fps):
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

def carregar_frame(cap, ff):
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

def calcular_brilho(frame):
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

def calcular_luminancia(frame):
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

def recuperar_valor_pixel(frame, x, y):
    """
    Pega valor do pixel em x,y de um frame 1D
    Entrada:
    - frame: frame com 2 dimensao
    Saida:
    - valor do pixel em  x,y
    """
    valor_pixel = frame[y, x]
    return valor_pixel

def calcular_distancia_focal(dist_cam_bat, dist_bal_bat,
                            val_bal_1, val_bal_2,
                            pix_bal_1, pix_bal_2):
    """
    Calcula o foco F
    Entrada:
    - dist_cam_bat: distancia da camera ao batedor, em metros
    - dist_bal_bat: ditancia da baliza ao batedor, em metros
    - val_bal_1: valor da baliza inferior, em metros
    - val_bal_2: valor da baliza superior, em metros
    - pix_bal_1: posicao do pixel da baliza inferior, em pixel
    - pix_bal_2 posicao do pixel valor da baliza superior, em pixel
    Saida:
    - F: distancia focal, em pixel?
    """

    # distancia entre as duas marcacoes da baliza, em metros
    dist_bal_mt = np.abs(val_bal_1 - val_bal_2)

    # distancia entre as duas marcacoes da baliza, em metros
    dist_bal_px = np.abs(pix_bal_1 - pix_bal_2)

    # distancia entre a camera e a baliza
    dist_cam_bal = dist_cam_bat - dist_bal_bat

    # calculo da distancia focal
    F = dist_bal_px * dist_cam_bal / dist_bal_mt

    return F, dist_cam_bal

def calcular_angulos(f, dist_cam_bal, pos_centro_ccd_px, pos_bal_1_px, alt_cam, alt_bal_nm, pix_cr):
    """
    Calcula altura de onda
    Entrada:
    f_inf: distancia focal da camera inferior
    pos_central_ccd_inf_px: posicao central do ccd em pixel
    pix_bal_1_inf
    alt_cam: altura da camera, em metros
    alt_bal_nm: altura da baliza ate o nivel medio, em metros
    pix_cr: indice do pixel na posicao da crista
    Saida:
    angulos em radianos
    """

    # angulo entre o zero da baliza e o centro do CCD da camera infeior
    ang_bal1_ccd = np.arctan((pos_centro_ccd_px - pos_bal_1_px) / f)

    # angulo entre a vertical e o zero da baliza (posicao inferior (1) da baliza)
    ang_vert_bal1 = np.arctan(dist_cam_bal / (alt_cam - alt_bal_nm))

    # angulo de picagem da camera
    ang_pitch = np.deg2rad(90 - np.rad2deg(ang_vert_bal1) - (-np.rad2deg(ang_bal1_ccd)))

    # angulo entre o centro do ccd e a posicao da crista em pixel
    ang_ccd_cr = np.arctan((pix_cr - pos_centro_ccd_px) / f)

    # angulo entre a vertical e a posicao da crista
    ang_vert_cr = np.deg2rad(90 - (np.rad2deg(ang_ccd_cr) + np.rad2deg(ang_pitch)))

    angs = ang_bal1_ccd, ang_vert_bal1, ang_pitch, ang_ccd_cr, ang_vert_cr

    return angs

def calcular_distancia_horizontal_proj_crista(alt_cam, ang_vert_cr):
    """
    Calcula distancias no plano horizontal na linha da agua
    Entrada:
    alt_cam: altura da camera
    alt_nominal_onda
    Saida:
    """

    # posicao da projecao da crista no plano da linha de agua (Xhi)
    dist_hor_proj_cr = np.tan(ang_vert_cr) * alt_cam

    return dist_hor_proj_cr

def calcular_distancias_horizontais(alt_cam, ang_vert_cr, alt_nominal_onda):
    """
    Calcula distancias no plano horizontal na linha da agua
    Entrada:
    alt_cam: altura da camera
    alt_nominal_onda: altura de onda gerada no batedor
    Saida:
    """

    # posicao da projecao da crista no plano da linha de agua (Xhi)
    dist_hor_proj_cr = np.tan(ang_vert_cr) * alt_cam

    # distancia horizontal ate a crista
    dist_hor_cr = - (np.tan(ang_vert_cr) * alt_nominal_onda / 2) + dist_hor_proj_cr

    return dist_hor_proj_cr, dist_hor_cr

# def calcular_altura_onda(alt_cam_inf=0.24, alt_cam_sup=0.48, dist_hor_proj_cr_inf=6.81, dist_hor_proj_cr_sup=5.93):
def calcular_altura_onda(alt_cam_inf, alt_cam_sup, dist_hor_proj_cr_inf, dist_hor_proj_cr_sup):
    """
    Calcula altura de onda por estereoscopia
    Entrada:
    alt_cam_inf: altura da camera inferior
    alt_cam_sup: altura da camera superior
    dist_hor_proj_cr_inf (XHi)
    dist_hor_proj_cr_sup
    Saida:
    altura de onda - H
    """

    # ampitude da onda
    amp = alt_cam_sup - (alt_cam_sup - alt_cam_inf) * (alt_cam_sup / dist_hor_proj_cr_sup) / ((alt_cam_sup / dist_hor_proj_cr_sup) - (alt_cam_inf / dist_hor_proj_cr_inf))

    # altura da onda
    H = amp * 2

    return H

def achar_horizonte(frame):
    pass

def achar_nivel_medio():
    pass

def achar_crista(img, ncristas, linf, lsup):
    """
    Acha uma matriz com as cristas.
    Entrada:
    ncristas - numero maximo de cristas a ser identificada
    ncf - numero de colunas do frame (resolucao horizontal)
    linf - limite inferior onde esta a agua
    lsup - limite superior onde esta a agua
    Saida:
    cristas - matriz com as cristas. A dimensao da matriz eh (ncrista,ncolunasframes)
    """


    # thres = 0.02 / np.max(c)
    thres = 0.4 # de 0 a 1
    min_dist = 70

    cols = []
    cristas = np.ones((ncristas, img.shape[1])) * np.nan

    # plt.close('all')
    # plt.figure(figsize=(12,8))
    # plt.imshow(img)

    for c in np.arange(0,img.shape[1]):
    # for c in range(0,1):

        vet1 = pd.Series(img[linf:lsup,c].astype(float))

        vet = vet1.rolling(window=7, center=True).mean()

        indexes = peakutils.indexes(vet, thres, min_dist)

        cols.append(indexes)

        cristas[:len(indexes),c] = indexes + linf
        # print (cristas[:len(indexes),c])


    # loop para correcao de reflexos das cristas
    for l in range(cristas.shape[0]):

        cr = cristas[l,:]

        # correcao 1
        # cristas[l, np.where(np.abs(cr) > cr.mean() + 10)[0]] = np.nan

        # correcao 2
        cristas[l, np.where(cr > np.nanmean(cr[:20]) + 10)[0]] = np.nan
        cristas[l, np.where(cr < np.nanmean(cr[:20]) - 10)[0]] = np.nan


        # plt.plot(vet+c, range(linf, lsup),'k')
        # plt.plot(vet[indexes]+c, np.arange(linf, lsup)[indexes],'.r')

    cristas = pd.DataFrame(cristas)
    # cristas = pd.DataFrame(cristas).interpolate(method='linear', axis=1).ffill()
    # cristas = cristas.rolling(window=10, center=True, axis=1).mean()


    return cristas


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

def plotar_figura(cont_frame, nfi1, nff1, eixo_x, serie_pixel, pathout):

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

def achar_posicao_px_crista(im):

    print ('clique na crista')
    plt.figure()
    plt.imshow(im)
    pxy = plt.ginput(1)
    Phi = pxy[0][1]
    plt.close('all')
    return Phi

def achar_posicao_0_baliza(im):

    print ('clique na posicao 0 da baliza')
    plt.figure()
    plt.imshow(im)
    pxy = plt.ginput(1)
    bpx = pxy[0][1]
    plt.close('all')
    return bpx

def achar_posicao_200_baliza(im):

    print ('clique na posicao 200 da baliza')
    plt.figure()
    plt.imshow(im)
    pxy = plt.ginput(1)
    Bpx = pxy[0][1]
    plt.close('all')
    return Bpx



if __name__ == '__main__':

    # dados de entrada

    # altura nominal da onda, em metros
    alt_nominal_onda = 0.1

    # distancia da baliza ao batedor
    dist_bal_bat = 7.5

    # distancia da camera ao batedor, em metros
    dist_cam_bat = 14.71

    # valor da medicao inferior e superior da baliza, em metros
    val_bal_1 = 0
    val_bal_2 = 0.2

    # altura das cameras superior e infeior com relacao ao nivel da agua, em metros
    alt_cam_inf = 0.24
    alt_cam_sup = 0.48
    alt_cams = [0.24, 0.48]

    # coordenada y do pixel da marcacao inferior (1) e superior (2) da baliza da camera inferior
    pix_bal_1_inf = 355
    pix_bal_2_inf = 200

    # coordenada y do pixel da marcacao inferior (1) e superior (2) da baliza da camera superior
    # pix_bal_1_sup = 355
    # pix_bal_2_sup = 200

    # altura do zero da regua (b) ate o nivel de agua, em metros
    alt_bal_nm = 0.172

    # coordenada y dos pixels das cristas
    # pix_cr_inf = 418
    # pix_cr_sup = None
    # pix_cr1_inf = 499
    # pix_cr2_inf = 421
    # pix_cr1_sup = 911
    # pix_cr2_sup = 755
    pix_cr_inf = [499, 421] # crista 1 e crista 2
    pix_cr_sup = [911, 755]

    # paths dos videos
    video_inf = os.path.join(os.environ['HOME'] + '/Documents/ondometro_videos/laboceano/CAM3/T100/', 'T100_570003_CAM3.avi')
    video_sup = os.path.join(os.environ['HOME'] + '/Documents/ondometro_videos/laboceano/CAM1/T100/', 'T100_570003_CAM1.avi')

    # path de saida das imagens
    # pathout = os.environ['HOME'] + '/Documents/teste9/'

    # numero do frame inicial e final do trecho do video a processar
    timei, timef = '00:52', '00:53'

    # chama funcoes

    # load video
    cap_inf, fps_inf = carregar_video(video_inf)
    cap_sup, fps_sup = carregar_video(video_sup)

    # find initial and final frames based on time MM:SS
    nfi_inf, nff_inf = achar_frame_inicial_e_final(timei, timef, fps_inf)
    nfi_sup, nff_sup = achar_frame_inicial_e_final(timei, timef, fps_sup)

    # read frames
    frame_inf = carregar_frame(cap_inf, ff=nfi_inf)
    frame_sup = carregar_frame(cap_sup, ff=nfi_sup)

    # converte para cinza
    im_inf = cv2.cvtColor(frame_inf, cv2.COLOR_BGR2GRAY)
    im_sup = cv2.cvtColor(frame_sup, cv2.COLOR_BGR2GRAY)

    ########### inicio dos calculos ###############

    # acha matrizes com as cristas (cada linha representa uma crista)

    # camera inf
    # cristas, cols, vet, indexes = achar_crista(img=im_inf, ncristas=20, linf=620, lsup=930)

    # camera sup
    cristas = achar_crista(img=im_sup, ncristas=20, linf=665, lsup=950)


    plt.figure()
    plt.imshow(frame_sup)
    plt.plot(cristas.iloc[0,:],'y', linewidth=1)
    plt.plot(cristas.iloc[1,:],'y', linewidth=1)
    plt.plot(cristas.iloc[2,:],'y', linewidth=1)
    plt.show()

    stop


    # ponto central da imagem (posição do centro do CCD em pixel)
    pos_central_ccd_inf_px = im_inf.shape[0] / 2
    pos_central_ccd_sup_px = im_sup.shape[0] / 2

    # # achar a posicao da crista em pixel
    # Phi1 = acha_posicao_px_crista(im1)
    # Phi2 = acha_posicao_px_crista(im2)

    # bpx1 = acha_posicao_0_baliza(im1)
    # bpx2 = acha_posicao_0_baliza(im2)

    # Bpx1 = acha_posicao_200_baliza(im1)
    # Bpx2 = acha_posicao_200_baliza(im2)

    # f_sup, dist_cam_bal = calcular_distancia_focal(dist_cam_bat=dist_cam_bat, dist_bal_bat=dist_bal_bat,
    #                                 val_bal_1=val_bal_1, val_bal_2=val_bal_2,
    #                                 pix_bal_1=pix_bal_1_sup, pix_bal_2=pix_bal_2_sup)




    # print ('Distancia focal camera inferior: {f_inf}'.format(f_inf=f_inf))
    # print ('Distancia focal camera superior: {f_sup}'.format(f_sup=f_sup))

    # loop para calcular as distancias de cada crista em cada camera

    # loop das cameras
    cont_alt_cam = 0
    for cam in [pix_cr_inf, pix_cr_sup]:

        f_inf, dist_cam_bal = calcular_distancia_focal(dist_cam_bat=dist_cam_bat, dist_bal_bat=dist_bal_bat,
                                val_bal_1=val_bal_1, val_bal_2=val_bal_2,
                                pix_bal_1=pix_bal_1_inf, pix_bal_2=pix_bal_2_inf)


        cont_alt_cam += 1
        if cont_alt_cam == 1:
            alt_cam = alt_cams[0]
        else:
            alt_cam = alt_cams[1]

        # loop de cristas
        for cr in range(len(cam)):

            angs = calcular_angulos(f=f_inf,
                                pos_centro_ccd_px=pos_central_ccd_inf_px,
                                pos_bal_1_px=pix_bal_1_inf,
                                alt_cam=alt_cam_inf,
                                alt_bal_nm=alt_bal_nm,
                                pix_cr=cam[cr])

            ang_bal1_ccd, ang_vert_bal1, ang_pitch, ang_ccd_cr, ang_vert_cr = angs


            dist_hor_proj_cr, dist_hor_cr = calcular_distancias_horizontais(alt_cam=alt_cam,
                                                                            ang_vert_cr=ang_vert_cr,
                                                                            alt_nominal_onda=alt_nominal_onda)

            print (dist_hor_cr)

    H = calcular_altura_onda()

    # dist_hor_proj_cr_inf, dist_hor_cr_inf = calcular_distancia_horizontal_proj_crista(alt_cam=alt_cam_inf,
    #                                                                                   ang_vert_cr=ang_vert_cr_inf)

    # cristas = {'crista_01_inf': list(angs_inf)+[dist_hor_proj_cr_inf, dist_hor_cr_inf],
    #            'crista_01_sup': list(angs_inf)+[dist_hor_proj_cr_inf, dist_hor_cr_inf]}


    # Distacia focal F = (bpx – Bpx)*7,21/0,20 (F em pixel)
    # F1 = (np.abs(bpx1 - Bpx1)) * (dist_cam_bat - dist_bal_bat) / bB
    # F2 = (np.abs(bpx2 - Bpx2)) * (dist_cam_bat - dist_bal_bat) / bB

    # # delta do centro da camera a crista
    # delta_C1 = Phi1 - P0i1
    # delta_C2 = Phi2 - P0i2

    # # Angulo entre a crista e o centro do CCD
    # dhi1 = np.arctan(delta_C1 / F1)
    # dhi2 = np.arctan(delta_C2 / F2)

    # # delta do centro da camera a posicao 200 da baliza
    # delta_B1 = Bpx1 - P0i1
    # delta_B2 = Bpx2 - P0i2

    # # Angulo entre o zero da baliza e o centro do CCD
    # dbali1 = np.arctan(Bpx1 / F1)
    # dbali2 = np.arctan(Bpx2 / F2)

    # # Angulo entre a vertical e o zero da baliza
    # alfai1 = np.arctan(XB / (C1 - b))
    # alfai2 = np.arctan(XB / (C2 - b))

    # # Angulo de picagem da câmera (Pitch) (feito uma vez so no inicio)
    # Yi1 = 90 - alfai1 - dbali1
    # Yi2 = 90 - alfai2 - dbali2
