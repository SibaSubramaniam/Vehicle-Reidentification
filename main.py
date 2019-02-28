from glob import glob
import cv2
from yolo_wrapper import Yolo

foldername='frame_set'
yolo=Yolo()
for i in glob(foldername+'/*.jpg'):
	print(i)
	yolo.run(i)
	