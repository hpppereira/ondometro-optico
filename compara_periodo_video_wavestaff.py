# Comparar espectro de brilho e espectro
# de elevacao do waveprobe
#
# Ultimas modificacoes:
# Henrique Pereira - 17/05/2022
#
# Infos Waveprobe:
# Fs = 60 Hz

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import mlab
plt.close('all')

if __name__ == "__main__":

    # -------------------------------------------------------------------------- #
    # leituras

    # paths
    pth_out = '/home/hp/gdrive/ondometro_optico/'
    pth_wavestaff = '/home/hp/shared/ufrjdrive/ondometro_data/wavestaffs/'

    fln = 'T100_570002'

    # leitura do arquivo de brilho
    brilho = pd.read_csv(pth_out + 'brilho_{}_CAM1.csv'.format(fln), index_col='time')

    # leitura do arquivo de waveprobe
    wstf = sio.loadmat(pth_wavestaff + '{}.gin.mat'.format(fln))

    # cria dataframe com indice do waveprobe
    wp = pd.DataFrame(wstf['WP_01'][:,0], columns=['WP_01'])
    wp.index = np.arange(0, len(wp)*(1./60), (1./60))
    wp.index.name = 'time'
    wp = wp / 1000.0

    # -------------------------------------------------------------------------- #
    # calculo dos espectros

    # brilho
    nfft_brilho = int(len(brilho)/2)
    fs_brilho = 30.0

    espec_brilho = mlab.psd(brilho.brilho.values,
                    NFFT=nfft_brilho,
                    Fs=fs_brilho,
                    detrend=mlab.detrend_mean,
                    window=mlab.window_hanning,
                    noverlap=int(nfft_brilho/2))

    f_brilho, sp_brilho = espec_brilho[1][1:], espec_brilho[0][1:]

    # waveprobe
    nfft_wp = int(len(wp)/2)
    fs_wp = 60.0

    espec_wp = mlab.psd(wp.WP_01.values,
                    NFFT=nfft_wp,
                    Fs=fs_wp,
                    detrend=mlab.detrend_mean,
                    window=mlab.window_hanning,
                    noverlap=int(nfft_wp/2))

    f_wp, sp_wp = espec_wp[1][1:], espec_wp[0][1:]

    # calculo do periodo de pico do waveprobe
    tp = 1.0 / f_wp[np.where(sp_wp == sp_wp.max())[0][0]]

    # -------------------------------------------------------------------------- #
    # plotagens

    # series de brilho e waveprobe
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    ax1.plot(brilho, color='blue')
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel('Brightness')
    ax1.set_xlim(50, 150)
    ax1.set_title(fln)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(wp, color='red')
    ax2.set_xlabel('Time [sec]')
    ax2.set_ylabel('Heave [m]')
    fig.tight_layout()
    fig.savefig(pth_out + 'series_{}.png'.format(fln), bbox_inches='tight')

    # espectros
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_ylabel('Brightness', color=color)
    ax1.plot(f_brilho, sp_brilho, '-', linewidth=4, markersize=2, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, 15)
    ax1.set_xticks(np.arange(15))
    ax1.set_title(fln + '\nPeak Period = {:.2f} sec'.format(tp))
    # ax1.grid()
    ax1.set_xlabel('Frequency [Hz]')
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Wave Probe [m²/Hz]', color=color)
    ax2.plot(f_wp, sp_wp, '-', markersize=2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(pth_out + 'espectros_{}.png'.format(fln), bbox_inches='tight')

    plt.show()







# ###############################################################################
# # calculo do periodo de onda



# # pega o tempo especifico de onde a onda ja chegou
# # df = df[20:]

# # calculo do espectro








# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.imshow(frame)
