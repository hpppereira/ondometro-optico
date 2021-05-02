import math

def waveheight():

    # DH é a distância entre as câmeras.

	DH = float(input('Distância entre as duas câmeras:'))

    # FD é a distância focal da câmera.
    # Entrará aqui, um parâmetro de calibração da câmera para calcular DF.

	FD = float(input('Distância focal da lente em pixels:'))

#	#------------->Input de dados
    #
    #	Neste script estou inserindo os dados manualmente.
    #	aqui as coordenadas dos pixels referentes à estes
    #	pontos deverá ser detectada automaticamente.
    #
#   #------------->Câmera superior:

	hsup1 = int(input('Posição do horizonte superior 1:'))
	hsup2 = int(input('Posição do horizonte superior 2:'))

	cristasup = int(
		input('Posição da crista na câmera superior:'))
	RPsup = int(
		input('Posição do ponto de referência na câmera superior:'))

#   #------------->Câmera inferior:

	hinf1 = int(input('Posição do horizonte inferior 1:'))
	hinf2 = int(input('Posição do horizonte inferior 2:'))

	cristainf = int(
		input('Posição da crista na câmera inferior:'))
	RPinf = int(
		input('Posição do ponto de referência na câmera inferior:'))

#   #------------>Triângulos semelhantes:
    #     Cada câmera tem 2 triângulos semelhantes, logo, dois catetos adjacentes
    # e tem o cateto oposto de valor igual. Podemos calcular 2 ângulos alpha para
    # a câmera superior e 2 ângulos beta para a câmera inferior.

#	#------------>Cálculo dos ângulos alpha da câmera superior:
	catsup1 = cristasup - hsup1
	catsup2 = RPsup - hsup2

	alpha1 = math.degrees(math.atan(catsup1 / FD))
	alpha2 = math.degrees(math.atan(catsup2 / FD))

#	#------------>Cálculo dos angulos beta da câmera inferior:
	catinf1 = cristainf - hinf1
	catinf2 = RPinf - hinf2

	beta1 = math.degrees(math.atan(catinf1 / FD))
	beta2 = math.degrees(math.atan(catinf2 / FD))

#	#------------>Cálculo da altura da onda:
    #
    #     Utilizando métodos trigonométricos podemos calcular a
    # altura da onda após calcular os comprimentos s1 e s2.
    #
    #     Temos 2 comprimentos horizontais (x1, x2) que equiva-
    # lem aos catetos opostos dos triângulos e são semelhantes
    # à distância focal.
    #https://pythonhelp.wordpress.com/2017/09/10/esquisitices-ou-nao-no-arredondamento-em-python-3/
    #    x1 --> das câmeras até a crista.
    #    x2 --> das câmeras até o ponto de referência.
    #-----------------------------------------------------------

	x1 = DH / (math.tan(math.radians(alpha1)) -
                   math.tan(math.radians(beta1)))
	x2 = DH / (math.tan(math.radians(alpha2)) -
                   math.tan(math.radians(beta2)))

	s1 = (math.tan(math.radians(beta1))) * x1
	s2 = (math.tan(math.radians(beta2))) * x2

	WH = s2 - s1

	print('A altura da onda é de %f cm' % round(WH, ndigits=2))
	
	print('A altura da onda é: ', round(WH, ndigits=3), 'cm')


waveheight()
