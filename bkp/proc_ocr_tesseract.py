# Reconhecimento de imagem
# Henrique Pereira
# Bruno Fonseca
# LIOc/UFRJ

# para instalar o tesseract
# sudo apt-get install tesseract-ocr
# sudo apt-get install tesseract

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pytesseract
plt.close('all')

# pathnames
pth = '/home/hp/gdrive/ondometro/videos/'
fln = 'T100_370002_CAM1.avi'
pth_fig = '/home/hp/Documents/ondometro/ocr/T100_370002_CAM1/frames/'
pth_out = '/home/hp/Documents/ondometro/ocr/T100_370002_CAM1/'

nome = fln.split('.')[0]

# leitura do video
cap = cv2.VideoCapture(pth + fln)

# lista com as datas retiradas dos frames
time_frames = []

#contador de frames
f = -1
while(cap.isOpened()):
    f += 1

    # leitura do frame
    ret, frame = cap.read()

    # verificar se frame existe
    if f == 100:
        frame = None
    if frame is None:
        break
    else:

        # retringe na imagem do tempo
        img = frame[952:1003,18:320,:]

        # converte tempo em string
        string = pytesseract.image_to_string(img)

        # cria string com tempo do video
        datastr = string[1:12]

        # print datastr
        print ('{} -- {}'.format(f, datastr))

        # adiciona string na lista de tempos
        time_frames.append(datastr)

        # plotagem da figura com o titulo
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        ax.imshow(frame)
        ax.set_title(datastr)
        fig.savefig('{}/{}'.format(pth_fig, nome+'_'+datastr+'.png'))
        plt.close('all')

# cria series com o time_frames
s = pd.Series(time_frames, name='time_ocr')
s.index.name = 'frame'

# salva arquivo em csv com os tempos dos frames do OCR
s.to_csv(pth_out + 'time_frames.csv')