# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

def process(img):
    #img = image.load_img(img_path, target_size=(224, 224))
    #img_data = image.img_to_array(img)
    #img_data = np.expand_dims(img, axis=0)
    #img_data = preprocess_input(img_data)

    img_data = np.expand_dims(img, axis=0)
    img_data=img_data/255.0

    feat = model.predict(img_data)
    
    return feat.flatten()
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
print(configPath,weightsPath)
#net=cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')

# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
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

		if LABELS[classID]!='car':
			continue
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"]:
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
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])


dist= []
count=1
# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		
		img=image[y:y+h,x:x+w]
		img=cv2.resize(img.astype('uint8'), (224,224))
		img_flat=process(img)
		dist.append(img_flat)
		filename='cam_2_car_'+str(count)+'.jpg'
		cv2.imwrite(filename,img)
		count+=1
		#cv2.imshow('img',img)
		#cv2.waitKey(0)
		
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		#cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
		#	0.5, color, 2)
dist_mat=np.asarray(dist)
print(dist_mat.shape)
q_img=cv2.imread('car_cam_1.jpg')
q_img=cv2.resize(q_img.astype('uint8'), (224,224))
q_img=process(q_img)

q_mat = np.expand_dims(q_img, axis=0)
print(type(q_mat))
print(q_mat.shape)

dist=np.sqrt(np.sum((q_mat-dist_mat)**2,axis=1))
i=np.argmin(dist)
(x, y) = (boxes[i][0], boxes[i][1])
(w, h) = (boxes[i][2], boxes[i][3])
cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)