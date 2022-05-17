"""
Ondometro - LabOceano

Summary:
Main Program to Process a video with two vertical cameras

Camera 1 (c1) - Inferior
Camera 2 (c2) - Superior

Input:
1. Pathname and filename of the two videos
2. Initial and final frame

Functions:
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
import peakutils
import warnings
warnings.filterwarnings("ignore")

plt.close('all')

def read_video(pathname):
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

def find_initial_and_final_frames(timei, timef, fps):
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

def read_frame(cap, ff):
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

def calculate_brigthness(frame):
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

def calculate_luminance(frame):
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

def calculate_focal_distance(dist_cam_bat, dist_bal_bat, val_bal_inf,
                             val_bal_sup, pix_bal_inf, pix_bal_sup):
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
    dist_bal_mt = np.abs(val_bal_inf - val_bal_sup)

    # distancia entre as duas marcacoes da baliza, em metros
    dist_bal_px = np.abs(pix_bal_inf - pix_bal_sup)

    # distancia entre a camera e a baliza
    dist_cam_bal = dist_cam_bat - dist_bal_bat

    # calculo da distancia focal
    f = dist_bal_px * dist_cam_bal / dist_bal_mt

    return f, dist_cam_bal

def find_max(frame, linf, lsup, win=8, limcr=4):
    """
    Acha maximos de brilho
    Entrada:
    - frame: frame com 2 dimensoes
    - linf: limite inferior para achar as cristas
    - lsup: limite superior para achar as cristas
    - win: janela para a media movel (default=8)
    - limcr: limite para deteccao de cristas (default=4)
    Saida:
    """

    # thres = 0.4 # de 0 a 1
    # min_dist = 70

    # vetor de zeros a serem acumuladas as cristas encontradas
    ccr = np.zeros(frame.shape[0])

    # loop para variar as colunas
    for c in range(frame.shape[1]):

        print (c)

        # vetor de coluna
        vet = pd.Series(frame[linf:lsup,c].astype(float))

        # media movel e derivada
        # numero de pontos para media movel
        mm = []
        for i in np.arange(win,len(vet)-win):
            a = vet[i-win:i]
            b = vet[i+1:i+win+1]
            inf = a.mean()
            sup = b.mean()
            mm.append(sup - inf)

        # vetor com a media movel
        mm = np.array([0]*win + mm + [0]*win)

        # acha valores maiores que um limite para identificacao de cristas
        icr = np.where(mm > limcr)[0] + linf

        # 'histograma' de distribuicao de maximos (pontos amarelos em y)
        ccr[icr] = ccr[icr] + 1

    return ccr

def find_creasts(img, ncristas, linf, lsup):
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
    cristas = pd.DataFrame(cristas).interpolate(method='linear', axis=1).ffill()
    cristas = cristas.rolling(window=10, center=True, axis=1).mean()
    cristas = pd.DataFrame(cristas).interpolate(method='linear', axis=1).ffill()

    # media das 1 primeiras cristas
    cristas_mean = np.array(cristas.iloc[:,:].mean(axis=1))

    # faz matriz de media
    cristas = pd.DataFrame(np.ones(cristas.shape) * cristas_mean[:,np.newaxis])

    return cristas[:3], cristas_mean[:3]

def calculate_angles(f, dist_cam_bal, pix_centro_ccd, pix_bal_inf, alt_cam,
                     alt_bal_nm, pix_cr):
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
    ang_bal_inf_ccd = np.arctan((pix_centro_ccd - pix_bal_inf) / f)

    # angulo entre a vertical e o zero da baliza (posicao inferior (1) da baliza)
    ang_vert_bal_inf = np.arctan(dist_cam_bal / (alt_cam - alt_bal_nm))

    # angulo de picagem da camera
    ang_pitch = np.deg2rad(90 - np.rad2deg(ang_vert_bal_inf) - (-np.rad2deg(ang_bal_inf_ccd)))

    # angulo entre o centro do ccd e a posicao da crista em pixel
    ang_ccd_cr = np.arctan((pix_cr - pix_centro_ccd) / f)

    # angulo entre a vertical e a posicao da crista
    ang_vert_cr = np.deg2rad(90 - (np.rad2deg(ang_ccd_cr) + np.rad2deg(ang_pitch)))

    # angs = ang_bal_inf_ccd, ang_vert_bal_inf, ang_pitch, ang_ccd_cr, ang_vert_cr

    return ang_bal_inf_ccd, ang_vert_bal_inf, ang_pitch, ang_ccd_cr, ang_vert_cr

def calculate_horizontal_distances_creast(alt_cam, ang_vert_cr, alt_nominal_onda):
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

def calculate_wave_height(alt_c1, alt_c2, dist_hor_proj_cr_c1, dist_hor_proj_cr_c2):
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
    # amp = alt_cam_sup - (alt_cam_sup - alt_cam_inf) * (alt_cam_sup / dist_hor_proj_cr_sup) / ((alt_cam_sup / dist_hor_proj_cr_sup) - (alt_cam_inf / dist_hor_proj_cr_inf))
    amp = alt_c2 - (alt_c2 - alt_c1) * (alt_c2 / dist_hor_proj_cr_c2) / ((alt_c2 / dist_hor_proj_cr_c2) - (alt_c1 / dist_hor_proj_cr_c1))

    # altura da onda
    alt_cr = amp * 2

    return alt_cr

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
    fig.savefig('{pathout}fig_{cont_frame}'.format(pathout=pathout,
                 cont_frame=str(cont_frame).zfill(3)))
    # plt.close('all')
    plt.show()
    return

def plot_creasts(frame_c1, frame_c2, cristas_c1, cristas_c2,
                 dist_hor_cr_c1, dist_hor_cr_c2, alt_cr, nframe, idx2, pathfig):

    plt.figure(figsize=(12,14))
    plt.subplot(211)
    plt.imshow(frame_c2)
    plt.plot(cristas_c2.iloc[idx2,100:300],'y', linewidth=.9)
    plt.title('Frame No: {:.0f} \n Alt. Crista {:.2f} m \n --- \n Camera Superior \n Dist. Creast: {:.2f} m'.format(nframe, alt_cr, dist_hor_cr_c2))
    plt.subplot(212)
    plt.imshow(frame_c1)
    plt.plot(cristas_c1.iloc[0,100:300],'y', linewidth=.9)
    plt.title('Camera Inferior \n Dist. Creast: {:.2f} m'.format(dist_hor_cr_c1))
    plt.savefig(pathfig + 'img_{}.png'.format(nframe), bbox_inches='tight')
    plt.close('all')
    return

def make_gif():
    string = "ffmpeg -framerate 10 -i %*.png output.mp4"
    os.system(string)
    return

if __name__ == '__main__':

    ############################################################################
    # dados de entrada
    ############################################################################

    # altura nominal da onda, em metros
    alt_nominal_onda = 0.1

    # periodo nominal da onda, em segundos
    per_nominal_onda = 1.4

    # comprimento nominal da onda, teoria linear
    comp_nominal_onda = 1.56 * per_nominal_onda ** 2

    # distancia da baliza ao batedor
    dist_bal_bat = 7.5

    # distancia da camera ao batedor, em metros
    dist_cam_bat = 14.71

    # valor da medicao inferior e superior da baliza, em metros
    val_bal_inf = 0
    val_bal_sup = 0.2

    # altura das cameras inferior e superior com relacao ao nivel da agua
    # em metros
    alt_c1 = 0.24
    alt_c2 = 0.48

    # altura do zero da regua ate o nivel de agua, em metros
    alt_bal_nm = 0.172

    # coordenada y do pixel da marcacao inferior e superior da baliza
    # camera inferior
    pix_bal_inf_c1 = 353
    pix_bal_sup_c1 = 196

    # camera superior
    pix_bal_inf_c2 = 710
    pix_bal_sup_c2 = 563

    # coordenada y dos pixels das cristas
    pix_cr_c1 = [499, 42] # crista 1 e crista 2
    pix_cr_c2 = [911, 755]

    # name of video (without extension and name)
    filename = 'T100_570003'
    pathname = os.environ['HOME'] + '/Documents/ondometro_videos/laboceano/'
    pathfig = os.environ['HOME'] + '/Documents/ondometro_results/{}/'.format(filename)

    # pathname do video inferior)
    video_c1 = os.path.join(pathname, '{}_CAM3.avi'.format(filename))

    # pathname video superior
    video_c2 = os.path.join(pathname, '{}_CAM1.avi'.format(filename))

    # create folder for figure output
    os.system('mkdir {}'.format(pathfig))

    # numero do frame inicial e final do trecho do video a processar
    timei, timef = '00:30', '00:40'

    # limite inferior e superior para procurar cristas (limite da agua)
    linf_c1, lsup_c1 = 400, 800
    linf_c2, lsup_c2 = 665, 950

    print (filename)

    ############################################################################
    # call functions
    ############################################################################

    # load video
    cap_c1, fps_c1 = read_video(video_c1)
    cap_c2, fps_c2 = read_video(video_c2)

    # find initial and final frames based on time MM:SS
    nfi_c1, nff_c1 = find_initial_and_final_frames(timei, timef, fps_c1)
    nfi_c2, nff_c2 = find_initial_and_final_frames(timei, timef, fps_c2)

    # time series of wave heights and creasts
    output = []

    # inicia loop dos frames
    cont = -1
    for nframe in np.arange(nfi_c1, nff_c1):
        cont += 1
        print (cont)

        # read frames
        frame_c1 = read_frame(cap_c1, ff=nframe)
        frame_c2 = read_frame(cap_c2, ff=nframe)

        # converte para cinza
        gray_c1 = cv2.cvtColor(frame_c1, cv2.COLOR_BGR2GRAY)
        gray_c2 = cv2.cvtColor(frame_c2, cv2.COLOR_BGR2GRAY)

        # ponto central da imagem (posição do centro do CCD em pixel)
        pix_centro_ccd_c1 = gray_c1.shape[0] / 2
        pix_centro_ccd_c2 = gray_c2.shape[0] / 2

        ################################################
        ################################################

        # ccr = find_max(gray_c1, linf=linf_c1, lsup=lsup_c1, win=8, limcr=4)
        # stop

        ################################################
        ################################################

        # acha matrizes com as cristas (
        # cristas - cada linha representa uma crista)
        # cristas_mean - valor medio das 3 primeiras cristas (batedor para camera)

        # camera inferior
        cristas_c1, cristas_mean_c1 = find_creasts(img=gray_c1, ncristas=10,
                                                   linf=linf_c1, lsup=lsup_c1)

        # camera superior
        cristas_c2, cristas_mean_c2 = find_creasts(img=gray_c2, ncristas=10,
                                                   linf=linf_c2, lsup=lsup_c2)

        # calcula distancia focal

        # camera inferior
        f_c1, dist_c1_bal = calculate_focal_distance(dist_cam_bat=dist_cam_bat,
                                                     dist_bal_bat=dist_bal_bat,
                                                     val_bal_inf=val_bal_inf,
                                                     val_bal_sup=val_bal_sup,
                                                     pix_bal_inf=pix_bal_inf_c1,
                                                     pix_bal_sup=pix_bal_sup_c1)

        f_c2, dist_c2_bal = calculate_focal_distance(dist_cam_bat=dist_cam_bat,
                                                     dist_bal_bat=dist_bal_bat,
                                                     val_bal_inf=val_bal_inf,
                                                     val_bal_sup=val_bal_sup,
                                                     pix_bal_inf=pix_bal_inf_c2,
                                                     pix_bal_sup=pix_bal_sup_c2)

        # calcular angulos para 3 cristas

        # valores de distancia projetada e distancia real
        distancias_c1 = []
        distancias_c2 = []

        # indice da distancia a ser escolhida (a mesma crista indo do batedor as cameras)
        # ind_cr = []

        # loop variando as 3 cristas
        for c in range(3):

            # camera inferior
            ang_bal_inf_ccd_c1, ang_vert_bal_inf_c1, ang_pitch_c1, ang_ccd_cr_c1, ang_vert_cr_c1 = \
            calculate_angles(f=f_c1,
                           dist_cam_bal=dist_c1_bal,
                           pix_centro_ccd=pix_centro_ccd_c1,
                           pix_bal_inf=pix_bal_inf_c1,
                           alt_cam=alt_c1,
                           alt_bal_nm=alt_bal_nm,
                           pix_cr=cristas_mean_c1[c])

            # camera superior
            ang_bal_inf_ccd_c2, ang_vert_bal_inf_c2, ang_pitch_c2, ang_ccd_cr_c2, ang_vert_cr_c2 = \
            calculate_angles(f=f_c2,
                           dist_cam_bal=dist_c2_bal,
                           pix_centro_ccd=pix_centro_ccd_c2,
                           pix_bal_inf=pix_bal_inf_c2,
                           alt_cam=alt_c2,
                           alt_bal_nm=alt_bal_nm,
                           pix_cr=cristas_mean_c2[c])

            # calcula distancias horizontais das cristas

            # camera inferior
            dist_hor_proj_cr_c1, dist_hor_cr_c1 = calculate_horizontal_distances_creast(alt_cam=alt_c1,
                                                                                        ang_vert_cr=ang_vert_cr_c1,
                                                                                        alt_nominal_onda=alt_nominal_onda)

            # camera superior
            dist_hor_proj_cr_c2, dist_hor_cr_c2 = calculate_horizontal_distances_creast(alt_cam=alt_c2,
                                                                                        ang_vert_cr=ang_vert_cr_c2,
                                                                                        alt_nominal_onda=alt_nominal_onda)

            # distancias horizontais das 3 cristas para as cameras 1 e 2
            distancias_c1.append([dist_hor_proj_cr_c1, dist_hor_cr_c1])
            distancias_c2.append([dist_hor_proj_cr_c2, dist_hor_cr_c2])

        # distancas das cristas em array (projetada e real)
        distancias_c1 = np.array(distancias_c1)
        distancias_c2 = np.array(distancias_c2)

        # indice da crista a ser mapeada
        # ind_cr.append(distancias_c1[0,1])

        # calculates the index of nearst value o cristas1
        idx2 = np.abs(distancias_c1[0,1] - distancias_c2[:,1]).argmin()

        # distancias horizontais reais e projetadas das cristas equivalentes
        dcr1, dpcr1 = distancias_c1[0,1], distancias_c1[0,0]
        dcr2, dpcr2 = distancias_c2[idx2,1], distancias_c2[idx2,0]

        # condicao para distancia entre as cristas maior ou menor que X numero de onda
        diff_dist_creasts = np.abs(dcr2 - dcr1)

        # desvio padrao da serie de linha de crista
        # std_cr_c1 = np.std(cristas_c1.iloc[0,:])
        # std_cr_c2 = np.std(cristas_c2.iloc[1,:])

        # imprime resultados
        print ('--- Frame Number: {:.0f} ---'.format(nframe))
        print ('Distancia Horizontal Cam. Inferior - crista: {:.2f} metros da camera'.format(dcr1))
        print ('Distancia Horizontal Cam. Superior - crista: {:.2f} metros da camera'.format(dcr2))

        # ------------------- condicao ------------------- #
        if diff_dist_creasts > 0.5:
            print (' ** Nao processa. diff_dist > 0.5 m')
        # elif  (dcr1 > 7.0 and dcr1 < 9.0):
        else:
            print ('** Processando..')

            # calculate wave height
            alt_cr = calculate_wave_height(alt_c1=alt_c1,
                                           alt_c2=alt_c2,
                                           dist_hor_proj_cr_c1=dpcr1,
                                           dist_hor_proj_cr_c2=dpcr2)

            print ('Altura da Onda: {:.2f} cm'.format(alt_cr*100))

            # cria serie temporal de alturas
            output.append([alt_cr, dcr1, dcr2])

            plot_creasts(frame_c1, frame_c2, cristas_c1, cristas_c2,
                         dcr1, dcr2, alt_cr, nframe, idx2, pathfig)
            # ------------------- condicao ------------------- #

    output = pd.DataFrame(output, columns=['alt','dcr1','dcr2'])
    output.to_csv('out/{}.csv'.format(filename))
