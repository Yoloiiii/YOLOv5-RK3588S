import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
	raise IOError("cant")
	
while True:
	rel,frame = cap.read()
	cv2.imshow('input',frame)
	c = cv2.waitKey(1)
	if c== 32:
		cv2.imwrite("p.jpg", frame)
		break
	elif c==27:
		cap.release()
		cv2.destoryAllWindows()
		break
		
cap.release()
cv2.destoryAllWindows()
