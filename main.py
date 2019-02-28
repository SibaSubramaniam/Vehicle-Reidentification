from glob import glob
import cv2
from yolo_wrapper import Yolo
import os

foldername=raw_input("Folder Name : ")
os.mkdir(foldername+'_extracted')
yolo=Yolo()
for i in glob(foldername+'/*.jpg'):
	print(i)
	yolo.run(i)
	