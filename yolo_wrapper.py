import numpy as np
import argparse
import time
import cv2
import os

yolo_path="yolo-coco"
w_confidence=0.8
w_threshold=0.3

class Yolo:
	def __init__(self):
		
		# load the COCO class labels our YOLO model was trained on
		labelsPath = os.path.sep.join([yolo_path, "coco.names"])
		self.LABELS = open(labelsPath).read().strip().split("\n")

		# initialize a list of colors to represent each possible class label
		#np.random.seed(42)
		#self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),dtype="uint8")

		# derive the paths to the YOLO weights and model configuration
		weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
		configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

		# load our YOLO object detector trained on COCO dataset (80 classes)
		print("[INFO] loading YOLO from disk...")
		self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

		# determine only the *output* layer names that we need from YOLO
		ln = self.net.getLayerNames()
		self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	def run(self,path):
		
		name=path[:-4]
		frame_name="frame_set_extracted/"+name.split("/",1)[1]+"_"

		image = cv2.imread(path)
		(H, W) = image.shape[:2]
		
		# construct a blob from the input image and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes and
		# associated probabilities

		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		self.net.setInput(blob)
		start = time.time()
		layerOutputs = self.net.forward(self.ln)
		end = time.time()

		print("[INFO] YOLO took {:.6f} seconds".format(end - start))

		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if self.LABELS[classID]!='car':
					continue
				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > w_confidence:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, w_confidence,
			w_threshold)


		dist= []
		count=1
		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				
				img=image[y:y+h,x:x+w]
				img=cv2.resize(img.astype('uint8'), (224,224))
				filename=frame_name+str(count)+'.jpg'
				print("writing file "+filename)
				cv2.imwrite(filename,img)
				count+=1

