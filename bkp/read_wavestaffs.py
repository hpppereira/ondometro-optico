'''
Leitura dos dados dos Wave Staffs do LabOceano

Henrique P. P. Pereira - pereira.henriquep@gmail.com

Projeto: Ondometro Optico
Local: LabOceano - UFRJ
Arquivos: .mat

Data: 2017/06/08
'''

#load libraries
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

#pathname of data
pathname = os.environ['HOME'] + '/Dropbox//Ondometro_Optico/data/laboceano/'
filename = 'T100_110002.gin.mat'

#list .mat files
list_mat = []
for f in os.listdir(pathname):
	if f.endswith('.mat'):
		list_mat.append(f)

# leitura dos dados em estrutura
d = sio.loadmat(pathname + filename)

# ver o que tem nos dados
print (d.keys())

# plot um array
plt.figure()
plt.plot(d['WP_31'])

plt.show()