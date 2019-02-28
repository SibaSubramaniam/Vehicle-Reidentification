from glob import glob
import cv2
from yolo_wrapper import Yolo
import os

foldername='frame_set_2'
os.mkdir(foldername+'_extracted')
yolo=Yolo()
for i in glob(foldername+'/*.jpg'):
	print(i)
	yolo.run(i)
	