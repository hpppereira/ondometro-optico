"""
Recupera informacoes do video
do celular do Victor, que sera realizadas as imagens

Infos:
Resolucao: (720 x 1280)
FPS: 30.771220521318003

Metadata
ffmpeg -i <videofile> -f ffmetadata metadata.txt
ou
exiftool filme_henrique.mp4 > metadata.txt
"""

import os
import cv2

pathname = os.environ['HOME'] + '/Dropbox/ondometro/data/'
filename = 'filme_henrique.mp4'

cap = cv2.VideoCapture(pathname + filename)

# while(cap.isOpened()):
    
ret, frame = cap.read()

res = frame.shape
fps = cap.get(cv2.CAP_PROP_FPS)

os.system("ffmpeg -i %s%s -f ffmetadata metadata.txt" %(pathname,filename))