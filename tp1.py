import cv2
import numpy as np
from win32api import GetSystemMetrics
import math as m

vec = []
image_reserva = cv2.imread("C:/Users/flabe/Desktop/2018-1/ICV/TP1/dados/pattern_0001.png")
def openImage():
	image = cv2.imread("C:/Users/flabe/Desktop/2018-1/ICV/TP1/dados/pattern_0001.png")
	print(image)
	new_img = rotateImage(image, 45)
	print(new_img)
	cv2.imshow('image',new_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def f2():
	#reading the image 
	image = cv2.imread("C:/Users/flabe/Desktop/2018-1/ICV/TP1/dados/pattern_0001.png")
	edged = cv2.Canny(image,100,200)
	#cv2.imshow("Edges", edged)
	#cv2.waitKey(0)
	 
	#applying closing function 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
	print(closed)
	resized_image = cv2.resize(closed, (GetSystemMetrics(0), GetSystemMetrics(1)) ) 
	#cv2.imshow("Closed", resized_image)
	#cv2.waitKey(0)
	
	(_, contours, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#finding_contours 
	"""for c in contours:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)"""
	
	#kernel = np.ones((5,5), np.uint8)
	#img_erosion = cv2.erode(closed, kernel, iterations=1)
	for c in contours:
		#peri = cv2.arcLength(c, True)
		#approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		rect = cv2.boundingRect(c)
		#print("RECT = ", rect)
		if(rect[2] > 70 and rect[3] < 40 and rect[2] < 300):
		#	print("ENTREI")
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
		if(rect[3] < 20 or rect[3] > 200):
			continue
		if(rect[2] > 70 or rect[2] < 40): 
			continue	
		#print(cv2.contourArea(c))
		x,y,w,h = rect
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		vec.append(rect)
	H, W = image.shape[:2]
	ratio = GetSystemMetrics(1)/float(H)
	resized_image = cv2.resize(image, (int(W*ratio), int(H*ratio)) )
	cv2.imshow("Show",resized_image)
	cv2.waitKey(0)  
	#print("VEC = ", vec)#VEC contendo todas bouding boxes
	cv2.destroyAllWindows()
	getBlack(vec, image_reserva, closed)
	#print("SIZE = ", len(contours))
	#putInVector(contours)
	#cv2.drawContours(image, contours, -1, (0,255,0), 3)
	#resized_image = cv2.resize(image, (GetSystemMetrics(0), GetSystemMetrics(1)) ) 
	#resized_image = cv2.resize(image, (int(W*ratio), int(H*ratio)) ) 
	#cv2.imshow("Output", resized_image)
	#cv2.waitKey(0)

def getBlack(vec, image, closed):
	vecV = []
	vecH_1 = []
	vecH_2 = []
	h = image.shape[0]
	w = image.shape[1]
	for v in vec:
		x,y,w,h = v
		if(x < 300):
			#print("X = ", v)
			vecV.append(v)
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		if(y < 600 and x < 750):
			cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
			vecH_1.append(v)
			#print("Y = ", v)		
		elif(y < 600 and x >= 750):
			cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
			vecH_2.append(v)
			#print("Y = ", v)		
	ex_number = 15
	for v in vecV:
		boxes = []
		vx, vy, vw, vh = v
		for h1 in vecH_1:
			h1x, h1y, h1w, h1h = h1
			for bb in vec:
				bbx, bby, bbw, bbh = bb
				if(abs(bbx - h1x) < 50 and abs(bby-vy) < 35):						
					#print("X = ", bbx)
					#print("y = ", bby)
					#print("hX = ", h1x)
					#print("hy = ", h1y)
					cv2.rectangle(image,(bbx,bby),(bbx+bbw,bby+bbh),(255,0,0),2)									
					box_tuple = (bbx, bby, bbw, bbh)
					boxes.append(box_tuple)
		H, W = image.shape[:2]
		ratio = GetSystemMetrics(1)/float(H)
		resized_image = cv2.resize(image, (int(W*ratio), int(H*ratio)) )
		#cv2.imshow("Show",resized_image)
		#cv2.waitKey(0)  
		#print(closed)
		#print("BOXES  = ", boxes)			
		soma = getResp(closed, boxes, ex_number, image)
		print(ex_number, ":", end="")
		if(soma == "BRANCO"):
			print("Branco")
		elif(soma == "NULO"):	
			print("Nulo")
		else:
			print(soma)
		ex_number = ex_number - 1

	ex_number = 30
	for v in vecV:
		boxes = []
		vx, vy, vw, vh = v
		for h2 in vecH_2:
			h2x, h2y, h2w, h2h = h2
			for bb in vec:
				bbx, bby, bbw, bbh = bb
				if(abs(bbx - h2x) < 50 and abs(bby-vy) < 35):						
					#print("X = ", bbx)
					#print("y = ", bby)
					#print("hX = ", h1x)
					#print("hy = ", h1y)
					cv2.rectangle(image,(bbx,bby),(bbx+bbw,bby+bbh),(255,0,0),2)									
					box_tuple = (bbx, bby, bbw, bbh)
					boxes.append(box_tuple)
		H, W = image.shape[:2]
		ratio = GetSystemMetrics(1)/float(H)
		resized_image = cv2.resize(image, (int(W*ratio), int(H*ratio)) )
		#cv2.imshow("Show",resized_image)
		#cv2.waitKey(0)  
		#print(closed)
		#print("BOXES  = ", boxes)			
		soma = getResp(closed, boxes, ex_number, image)
		print(ex_number, ":", end="")
		if(soma == "BRANCO"):
			print("Branco")
		elif(soma == "NULO"):	
			print("Nulo")
		else:
			print(soma)
		ex_number = ex_number - 1	


def getResp(image, boxes, ex_number, image2):
	#print("BX = ", boxes[0][0])
	#cv2.imshow("IM", image)
	Resp_vec = []
	#print("COORD X = ", boxes[0][0])
	#print("COORD X2 = ", boxes[0][0] + boxes[0][2])
	#print("COORD Y = ", boxes[0][1])
	#print("COORD Y2 = ", boxes[0][1] + boxes[0][3])
	#cv2.rectangle(image,(boxes[0][0],boxes[0][1]),(bbx+bbw,bby+bbh),(255,0,0),2)									
	#image[boxes[0][1]:(boxes[0][1]+boxes[0][3]),boxes[0][0]:(boxes[0][0]+boxes[0][2])] = 100
	#image2 = image
	e = image[boxes[0][1]:(boxes[0][1]+boxes[0][3]),boxes[0][0]:(boxes[0][0]+boxes[0][2])]
	d = image[boxes[1][1]:(boxes[1][1]+boxes[1][3]),boxes[1][0]:(boxes[1][0]+boxes[1][2])]
	c = image[boxes[2][1]:(boxes[2][1]+boxes[2][3]),boxes[2][0]:(boxes[2][0]+boxes[2][2])]
	b = image[boxes[3][1]:(boxes[3][1]+boxes[3][3]),boxes[3][0]:(boxes[3][0]+boxes[3][2])]
	a = image[boxes[4][1]:(boxes[4][1]+boxes[4][3]),boxes[4][0]:(boxes[4][0]+boxes[4][2])]
	
	count_e = np.count_nonzero(e)
	Resp_vec.append(count_e)
	count_d = np.count_nonzero(d)
	Resp_vec.append(count_d)
	count_c = np.count_nonzero(c)
	Resp_vec.append(count_c)
	count_b = np.count_nonzero(b)
	Resp_vec.append(count_b)
	count_a = np.count_nonzero(a)
	Resp_vec.append(count_a)
	
	image2[boxes[0][1]:(boxes[0][1]+boxes[0][3]),boxes[0][0]:(boxes[0][0]+boxes[0][2])] = 200
	image2[boxes[1][1]:(boxes[1][1]+boxes[1][3]),boxes[1][0]:(boxes[1][0]+boxes[1][2])] = 200
	image2[boxes[2][1]:(boxes[2][1]+boxes[2][3]),boxes[2][0]:(boxes[2][0]+boxes[2][2])] = 200
	image2[boxes[3][1]:(boxes[3][1]+boxes[3][3]),boxes[3][0]:(boxes[3][0]+boxes[3][2])] = 200
	image2[boxes[4][1]:(boxes[4][1]+boxes[4][3]),boxes[4][0]:(boxes[4][0]+boxes[4][2])] = 200
	
	#for i in Resp_vec:
	#	print("I = ", i)
	#print()
	#print()	
	#print("RESP = ", Resp_vec)
	soma = sum(i > 1000 for i in Resp_vec)

	#print("Numero = ", ex_number, " Soma = ", soma)
	#print("count_d = ", count_d)
	#count = np.count_nonzeros(d)

	H, W = image.shape[:2]
	ratio = GetSystemMetrics(1)/float(H)
	resized_image = cv2.resize(image2, (int(W*ratio), int(H*ratio)) )
	cv2.imshow("Show",resized_image)
	cv2.waitKey(0)  

	if(soma == 1):
		for i in range(0, len(Resp_vec)):
			if(Resp_vec[i] > 1000):
				val = ord('A')
				val = val + (len(Resp_vec) - i - 1)
				return chr(val)
	elif(soma > 1):
		return "NULO"			
	else:	
		return "BRANCO"

	


def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def subimage(image, center, theta, width, height):

   ''' 
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = image.shape[:2]

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image

#openImage()

image = cv2.imread("C:/Users/flabe/Desktop/2018-1/ICV/TP1/dados/pattern_0001_scan.png")
#image = subimage(image, center=(500, 60), theta=30, width=1000, height=2000)
#cv2.imwrite('patch.jpg', image)
#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

f2()