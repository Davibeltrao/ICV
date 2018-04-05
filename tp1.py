import cv2
import numpy as np
from win32api import GetSystemMetrics
import math as m

def getV(vec):
	vecV = []
	vecSup = vec.copy()
	vecSup.sort(key=lambda x:x[0])
	vecV = vecSup[:15]
	vecV.sort(key=lambda x:x[1])
	return vecV

def getH(vec):
	vecH = []
	vecSup = vec.copy()
	vecSup.sort(key=lambda x:x[1])#ordena de acordo com o y
	vecH = vecSup[:12]#coordenadas horizontais
	vecHSup = vecH.copy()
	vecHSup.sort(key=lambda x:x[0])#ordena de acordo com o x
	vecH1 = vecHSup[:6]#primeira metade
	vecH2 = vecHSup[6:]#segunda metade
	vecH1.sort(key=lambda x:x[0])
	vecH2.sort(key=lambda x:x[0])
	return vecH1, vecH2

def Image_reader():
	vec = []
	#reading the image 
	image = cv2.imread("C:/Users/flabe/Desktop/2018-1/ICV/TP1/dados/pattern_0001_scan.png")
	image_reserva = image.copy()
	edged = cv2.Canny(image,100,200)
	#cv2.imshow("Edges", edged)
	#cv2.waitKey(0)
	 
	#criando imagem Branco-Preta
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	
	(_, contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	#TENTATIVA DE APLICAR EROSAO PARA MELHORAR A LEITURA DOS DADOS QUE CONTEM MARCACOES BORRADAS
		#kernel = np.ones((5,5), np.uint8)
		#img_erosion = cv2.erode(closed, kernel, iterations=1)

	#Desenhando contornos na imagem e armazenando BBs	
	for c in contours:
		rect = cv2.boundingRect(c)
		if(rect[2] > 70 and rect[3] < 40 and rect[2] < 300):
			x,y,w,h = rect
			w = m.ceil(w/2)
			x2 = x+h*2
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
			cv2.rectangle(image,(x2,y),(x2+w,y+h),(255,0,0),2)
			tup1 = (x,y,w,h)
			tup2 = (x2,y,w,h)
			vec.append(tup1)
			vec.append(tup2)
			continue
		if(rect[3] < 20 or rect[3] > 200):#descarta contornos com uma altura muito grande ou muito pequena
			continue
		if(rect[2] > 200 or rect[2] < 40):#descarta contornos com comprimento mto grande ou muito pequeno
			continue	
		#print(cv2.contourArea(c))
		x,y,w,h = rect
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		vec.append(rect)
	showImage(image)
	#print("VEC = ", vec)#VEC contendo todas bouding boxes
	cv2.destroyAllWindows()
	CalculateAnswer(vec, image_reserva, closed)
	
def CalculateAnswer(vec, image, closed):
	h = image.shape[0]
	w = image.shape[1]
	ex_number = 1

	vecV = getV(vec)# Retorna os contornos pretos verticais
	vecH_1, vecH_2 = getH(vec)# Retorna os contornos pretos horizontais(divide entre horizontal esquerda e horizontal direita)
	
	for i in vecV:
		x,y,w,h = i
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	showImage(image)

	for i in vecH_1:
		x,y,w,h = i
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
		showImage(image)

	for i in vecH_2:
		x,y,w,h = i
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		showImage(image)
	
	#itera para a primeira metade do gabarito
	for v in vecV:
		boxes = []
		vx, vy, vw, vh = v
		for h1 in vecH_1:
			h1x, h1y, h1w, h1h = h1
			for bb in vec:
				bbx, bby, bbw, bbh = bb
				if(abs(bbx - h1x) < 25 and abs(bby-vy) < 10):						
					cv2.rectangle(image,(bbx,bby),(bbx+bbw,bby+bbh),(0,255,0),2)									
					box_tuple = (bbx, bby, bbw, bbh)
					boxes.append(box_tuple)
					H, W = image.shape[:2]
					ratio = GetSystemMetrics(1)/float(H)
					resized_image = cv2.resize(image, (int(W*ratio), int(H*ratio)) )
					cv2.imshow("Show",resized_image)
					cv2.waitKey(0)  
		soma = getResp(closed, boxes, ex_number, image)
		print(ex_number, ":", soma)
		ex_number = ex_number + 1

	#itera para a segunda metade do gabarito
	for v in vecV:
		boxes = []
		vx, vy, vw, vh = v
		for h2 in vecH_2:
			h2x, h2y, h2w, h2h = h2
			for bb in vec:
				bbx, bby, bbw, bbh = bb
				if(abs(bbx - h2x) < 25 and abs(bby-vy) < 10):						
					cv2.rectangle(image,(bbx,bby),(bbx+bbw,bby+bbh),(255,0,0),2)									
					box_tuple = (bbx, bby, bbw, bbh)
					boxes.append(box_tuple)
					H, W = image.shape[:2]
					ratio = GetSystemMetrics(1)/float(H)
					resized_image = cv2.resize(image, (int(W*ratio), int(H*ratio)) )
					cv2.imshow("Show",resized_image)
					cv2.waitKey(0)  
		soma = getResp(closed, boxes, ex_number, image)
		print(ex_number, ":", soma)
		ex_number = ex_number + 1	


def getResp(image, boxes, ex_number, image2):
	Resp_vec = []
	
	a = image[boxes[0][1]:(boxes[0][1]+boxes[0][3]),boxes[0][0]:(boxes[0][0]+boxes[0][2])]
	b = image[boxes[1][1]:(boxes[1][1]+boxes[1][3]),boxes[1][0]:(boxes[1][0]+boxes[1][2])]
	c = image[boxes[2][1]:(boxes[2][1]+boxes[2][3]),boxes[2][0]:(boxes[2][0]+boxes[2][2])]
	d = image[boxes[3][1]:(boxes[3][1]+boxes[3][3]),boxes[3][0]:(boxes[3][0]+boxes[3][2])]
	e = image[boxes[4][1]:(boxes[4][1]+boxes[4][3]),boxes[4][0]:(boxes[4][0]+boxes[4][2])]
	
	count_a = np.count_nonzero(a)
	Resp_vec.append(count_a)
	count_b = np.count_nonzero(b)
	Resp_vec.append(count_b)
	count_c = np.count_nonzero(c)
	Resp_vec.append(count_c)
	count_d = np.count_nonzero(d)
	Resp_vec.append(count_d)
	count_e = np.count_nonzero(e)
	Resp_vec.append(count_e)
	
	#So coloca um quadrado cinza na posicao da linha atual
	image2[boxes[0][1]:(boxes[0][1]+boxes[0][3]),boxes[0][0]:(boxes[0][0]+boxes[0][2])] = 200
	image2[boxes[1][1]:(boxes[1][1]+boxes[1][3]),boxes[1][0]:(boxes[1][0]+boxes[1][2])] = 200
	image2[boxes[2][1]:(boxes[2][1]+boxes[2][3]),boxes[2][0]:(boxes[2][0]+boxes[2][2])] = 200
	image2[boxes[3][1]:(boxes[3][1]+boxes[3][3]),boxes[3][0]:(boxes[3][0]+boxes[3][2])] = 200
	image2[boxes[4][1]:(boxes[4][1]+boxes[4][3]),boxes[4][0]:(boxes[4][0]+boxes[4][2])] = 200
	
	#verifica qual a quantidade de pixels diferentes de zero(255) e se for alta, quer dizer que foi marcada a alternativa
	#1000 significa 1000 pixels iguais a 255
	soma = sum(i > 1000 for i in Resp_vec)

	showImage(image2)

	#retorna resposta
	if(soma == 1):
		for i in range(0, len(Resp_vec)):
			if(Resp_vec[i] > 1000):
				val = ord('A')
				val = val + i
				return chr(val)
	elif(soma > 1):
		return "NULO"			
	else:	
		return "BRANCO"

def showImage(image):
	H, W = image.shape[:2]
	ratio = GetSystemMetrics(1)/float(H)
	resized_image = cv2.resize(image, (int(W*ratio), int(H*ratio)) )
	cv2.imshow("Show",resized_image)
	cv2.waitKey(0)  


"""def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result"""


Image_reader()