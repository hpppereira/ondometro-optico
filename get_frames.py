# -*- coding: utf-8 -*-
"""
Retirar frames de um filme

- Filme_02

Cel_Victor (camera inferior)
20180405_165833.mp4
tempo_ini = 3:13 
tempo_fim = 3:26

tempo = 1.41 (joao com braço direito levantado)
frame_crista: frame_00:01:41.42.0000_015.png
frame_espraiamento: frame_00:01:35.000000_001.png

----------------------------------------------------

Cel_Nelson (camera superior)
20180405_165638_rot.mp4
tempo_ini = 2:00 
tempo_fim = 5:00
* necessario rotacionar 180 graus

tempo = 3.36 (joao com braço direito levantado)
frame_crista: frame_00:03:36.7200000_025.png
frame_espraiamento: frame_00:03:30.270000_010.png


"""

import os
import pandas as pd

make_get_metadata = False
make_get_frames = True

# ----------------------------------------------------------------------------------------------- #

# framedir = 'frames'
pathname = os.environ['HOME'] + '/Documents/ondometro/data/praia_seca_20180405/Cel_Nelson/filme_02/'
# pathname = os.environ['HOME'] + '/Documents/ondometro/data/praia_seca_20180405/Cel_Victor/filme_02/'
# framedirlap = 'frames_laplacian'
# filename = '20180405_165833.mp4'  # list of files
filename = '20180405_165638_rot.mp4'
framei = '00:03:30.000'  # first frame
# framei = '00:00:04.000'  # first frame
framef = '00:03:31.000'  # last frame
freq = '30ms'  # fps
fileout = 'frame'  # filename

def get_metadata(pathname,  filename):

    os.system('cd ' + pathname + '\n' + 
              "ffmpeg -i %s%s -f ffmetadata metadata.txt" %(pathname,filename))

def get_frames(pathname, filename, framei, framef, freq, fileout):

    """
    Get frames from a video
    :param pathname:
    :param moviefile:
    :param framei:
    :param framef:
    :param freq:
    :param fileout:
    :return: saved frames
    """

    frames1 = pd.date_range(framei, framef, freq=freq)

    def function1(x):
        return x.strftime('%H:%M:%S.%f')

    frames = frames1.format(formatter=function1)

    cont = 0
    for frame in frames:
        cont += 1
        os.system('cd ' + pathname + '\n' +
                  'ffmpeg -i ' + filename + ' -ss ' + frame +
                  ' -f image2 -vframes 1 ' + pathname + 'frames/'
                  + fileout + '_' + frame + '_' + str(cont).zfill(3) + '.png')

        # stop

# ----------------------------------------------------------------------------------------------- #

if make_get_frames:

	get_frames(pathname, filename, framei, framef, freq, fileout)

# if make_get_metadata:

#     get_metadata(pathname, filename)