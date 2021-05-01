# coding: utf-8
'''
author: matheus vieira - aug.2018
ubuntu 18.04 - python 3 ffmpeg 4.0.2

Project: WASS

Program to sync videos, using praat (audio cross correlation),ffmpeg, Handbrake and python

needs:
	sudo apt install handbrake-cli
	sudo apt-get install praat
		file: crosscorrelate.praat

do:
	- encoding videos : set fps of the video and change to constante frame rate (uses Handbrake)
	- find all .mp4 files in current directory, put all filenames in a list
	- Select one filename in the list to be the reference file and remove it from the list
	- Extract the audio of the reference file (ref.wav)
	- For each file in file list, cross-correlate its audio with ref.wav to find the offset (uses Praat).
		Add all results (filename, offset) to a list, which contains one item already: (ref filename, 0)
	- For each item in the results list, trim the video at the given start and duration, taking into
		account the offset. The reference file will be trimmed at exactly the start time (its offset is
		zero), but each other file will be trimmed from a start that is shifted by the respective offset.
	- Optionally convert the video files to the desired format

ref: http://www.dsg-bielefeld.de/dsg_wp/wp-content/uploads/2014/10/video_syncing_fun.pdf
	 https://handbrake.fr/docs/en/latest/cli/command-line-reference.html

# cam1 = Galaxy J1 Mini Prime BRANCO duos 31 fps - 5MP - res:1280x720
# cam2 = Galaxy J1 Mini BEGE			  30 fps - 5MP - res:1280x720
'''

import os, subprocess
from glob import glob

# change to video path
os.chdir("/home/matheus/projects/wass/sync/")

# enconding video file using HandBrake
# 	HandBrakeCLI [options] -i <source> -o <destination>
# 	set fps (-r), constante frame rate (--cfr) and bitrate (-b)
# 	CFR,VFR(default) or PFR(default with -r specified)

command = "HandBrakeCLI -r 15 --cfr -b 6000 -i video1.mp4 -o video1_alt.mp4"
command2 = "HandBrakeCLI -r 15 --cfr -b 6000 -i video2.mp4 -o video2_alt.mp4"

os.system(command)
os.system(command2)

#extract audio from video
clip_list = glob('*alt.mp4')
ref_clip_index = 0 # first clip used as reference
ref_clip = clip_list[ref_clip_index]
clip_list.pop(ref_clip_index) #remove the reference clip from the list

# extract the reference audio, which is the audio of the reference clip
command = "ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}".format(ref_clip,"ref.wav")
os.system(command)

results = []
results.append((ref_clip, 0)) #the reference clip has an offset of 0
for clip in clip_list:
	clipfile = clip.split(".")[0] + ".wav"
	command = "ffmpeg -i {0} -vn -acodec pcm_s16le -ar 44100 -ac 2 {1}".format(clip,clipfile)
	os.system(command)

	# calculate cross correlation using praat
	command = "praat /home/matheus/projects/wass/sync/crosscorrelate.praat ref.wav {}".format(clipfile)
	result = subprocess.check_output(command, shell=True)
	results.append((clip, result.decode("utf-8").split("\n")[0])) #result.split("\n")[0]
	# results.append((clip, 0.678)) # this value comes from praat correlation between 2 .wav files

#now that we got the results, delete all the WAV files, we don't need them
command = "rm *wav"
os.system(command)

# Trim the clips
for result in results:
	clip_start = 0	#the start of the trimmed part (for the reference clip)
	clip_dur = 720	#the duration of the trimmed part (both are in seconds)
	in_name = result[0]
	out_name = in_name.split('.')[0] + "_part.mp4"
	offset = round(float(result[1]),3) 	# offset = 0.678
	clip_start += offset

	#cutting and coverting to mp4
	command = "ffmpeg -i {0} -ss {1} -to {2} -qscale 0 {3} ".format(in_name,str(clip_start),str(clip_dur),out_name)
	# command = "ffmpeg -i {0} -b:v 6000k -ss {1} -to {2} {3} ".format(in_name,str(clip_start),str(clip_dur),out_name)
	# birate = 6000kbps
	# ss = tempo inicial / to = tempo final
	# qscale 0 = manter qualidade do video apos o recorte
	# a funcao .format associa o que esta dentra da funcao com os os {}
	# 1:in_name, 2: clipstart e clipdur, 3: output
	os.system(command)

# move sync and trimmed files to specific path
# command = "mv video1_alt_part.mp4 cam1_alt/"
# command2 = "mv video2_alt_part.mp4 cam2_alt/"
# os.system(command)
# os.system(command2)
