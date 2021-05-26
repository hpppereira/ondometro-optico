

# linha do inicio do ondometro optico no scripy ScreenPixelProfile.pas: 1216

# calcular brilho

# Intensity := (RGB.rgbtred+RGB.rgbtgreen+RGB.rgbtblue) DIV 3;
# Luminosity := (RGB.rgbtred+RGB.rgbtgreen+RGB.rgbtblue) DIV 3;

# //Cálculo de Luminância a partir das Primárias RGB
# //Luminosity := (2125*RGB.rgbtred+7154*RGB.rgbtgreen+721*RGB.rgbtblue) DIV 10000;
# //Cálculo de Luma a partir das Componentes não linears R’G’B’
# //Luminosity := (299*RGB.rgbtred+587*RGB.rgbtgreen+114*RGB.rgbtblue) DIV 1000;



# // CALCULO DOS GRADIENTES VERTICAIS PARA DETERMINAÇÃO DO HORIZONTE E DA INCLI/DECLINAÇÃO DA CAM!

# // Parâmetros da Câmera
# //AbCam := 40.4*PI/180.;   // Abertura Angular da Camera
# AbCam :=  StrToFloat(Edit_AbCam.Text)*PI/180.; //AbCam escrita na FORM.
# disth := (Bitmap.Height/2)/tan(AbCam/2); // disth é a distancia em pixels do plano do CCD ao ponto de convergência.

# Incli :=  0. ;
# DecliCam :=  0. ;


# DERI1V := AverLum[4]-AverLum[0];

# // determina a menor derivada. Utilizada no caso de nenhum ponto
# // ao longo da coluna satisfazer o critério de horizonte.

# // Parâmetros para ajuste da reta do Horizonte


# // FIM DO CALCULO DA 1ª DERIVADA VERTICAL PARA ONDÓGRAFO!!!!



# // Ajustando a Linha do Horizonte p/ correção de montanhas etc.

# //neste ponto já temos os "melhores" contrastes p/ cada coluna.

# //SELECIONA 10 PONTOS INICIAIS. má ideia esse 10!.

# ZN1[IX13] := ZZ1[IX13+ORIGEN];  //seleciona 10  contrastes.

# //AJUSTA UMA LINHA RETA E VERIFICA SE TODOS OS PONTOS FORAM APROVEITADOS

# //ESTRITA_MQUAD(ZN1,AN1,NI1,PN1,TemPontoRuim,TemReta); //  MINIMOS QUADRADOS ESTRITOS

# //SALVA A RETA

# //ACABOU DE SELECIONAR PONTOS EM CIMA DE RETAS ESTRITAS!



# // Atualiza os valores do ajuste

# // Linha do Horizonte ajustada

# // Parâmetros da Câmera

# //nh é a distância vertical do horizonte,  até o meio da tela.

# nh := (Bitmap.Height/2) - (A*(Bitmap.Width/2) + B )  ;

# // Inclinação da Câmera

# //IncliCam := - ArcTan(nh*tan(AbCam/2)/(Bitmap.Height/2));

# IncliCam :=  ArcTan(nh/disth);

# // Declinação da Câmera

# DecliCam := ArcTan(A);

# // H é a altura do CCD.
# H := Bitmap.Height-1;
# // W é a largura do CCD.
# W := Bitmap.Width-1;
# Incli :=  +IncliCam ;


# // Correndo a Imagem Horizontalmente para cálculo do Fluxo Óptico e FFT.

# ///////// INÍCIO DO CÁLCULO DAS DERIVADAS HORIZONTAL E VERTICAL PARA DIREÇÃO DE ONDA //////////////////////

# // As luminosidades Medias são médias ponderadas usando 9 pixels

# // AS VARIÁVEIS DERIH E DERIV (COM "I") SÃO USADAS PARA CALCULAR O GRADIENTE DA LUMINOSIDADE (VETORIAL) //

# // AS VARIÁVEIS DERH E DERV (SEM "I') SÃO USADAS PARA CALCULAR A DIREÇÃO DAS ONDAS (+90º > PHI > -90º) //

# // REPRESENTANDO O PERFIL DE ONDA NA IMAGEM!!  EXTINTO!! Henrique 05/05/2012

# // OBS.: AS DERIVADAS SÃO SEMPRE REPRESENTADAS PARA O PIXEL 6 LINHAS ACIMA

# teta_dir := ArcTan2(DERIV,DERIH);  // PARA GRADIENTE (COM "I"!)

# GRAD := SQRT(DERIH*DERIH + DERIV*DERIV);

# // Função REDUZ - reduz o âng ao 1º (pos) e 4º (neg) quadrantes. (ref à Latitude)

# teta_dr := ArcTan2(DERV,DERH);    // PARA DIREÇÃO DE ONDA (SEM "I"!)

# disp_dr := disp_dr + (teta_dr*teta_dr);  // SOMA DOS QUADRADOS PARA O CÁLCULO DE DISPERSÃO DO ANGULO (P/ DIREÇÃO DE ONDA!)

# dispH := dispH + (DERIH*DERIH);  // SOMA DOS QUADRADOS PARA O CÁLCULO DE DISPERSÃO (P/ GRADIENTE!, COM "I")

# dispV := dispV + (DERIV*DERIV);

# ///////// FINAL DO CÁLCULO DAS DERIVADAS HORIZONTAL E VERTICAL PARA DIREÇÃO DE ONDA /////////////////////

# ///////////// CRIANDO E ESCREVENDO MATRIZES COM AS LINHAS E COLUNAS DE PNTOS NA DIAGONAL... /////////////

# /////////////    ... DO FRAME PARA ANÁLISE DE FOURIER DE KX E KY           //////////////////////////////

# ////////// CALCULANDO AS DERIVADAS MEDIAS, H e V, O ANGULO, O GRADIENTE MÉDIO NO FRAME E AS DISPERSÕES ///////////////
