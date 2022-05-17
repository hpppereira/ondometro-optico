'''
author: matheus vieira - aug. 2018
ubuntu 18.04 py3 OpenCV 3.4.1

Project: WASS

Program to get frames of a video using OpenCV
do: get all frames from a video file and save in pathname_fig

ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

# cam1 = Galaxy J1 Mini Prime BRANCO duos 31 fps - 5MP - res:1280x720
# cam2 = Galaxy J1 Mini BEGE			  30 fps - 5MP - res:1280x720
'''
import numpy as np
import cv2
import os

a=12   #1 and 2 means the camera pathname suffix
for i in str(a):

    pathname        = os.environ['HOME'] + '/projects/wass/sync/cam'+i+'_alt/'
    pathname_fig    = os.environ['HOME'] + '/projects/wass/tests/sync_time/frames'+i+'/'
    filename        = 'video'+i+'_alt_part.mp4'

    # capture video object
    cap = cv2.VideoCapture(pathname + filename)
    #read frames
    ret, frame = cap.read()

    count = 0
    ret = True
    while ret:
        # rotate frame
        # frame=cv2.flip(frame,0) # flip frame vertically
        # frame=cv2.flip(frame,1) # flip frame honrizontally

        # save frames
        cv2.imwrite(pathname_fig + "000%d_01.tif" % count, frame)   # save frame as .tif
        ret, frame = cap.read()
        print ('make a new frame: ', ret)
        count += 1
