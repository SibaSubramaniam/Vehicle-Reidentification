from glob import glob
import cv2



foldername='frame_set'

for i in glob(foldername+'/*.jpg'):
	print(i)
	img=cv2.imread(i)
	cv2.imshow('img',img)
	cv2.waitKey(0)