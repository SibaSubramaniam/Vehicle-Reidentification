from ReID_CNN.Model_Wrapper import ResNet_Loader
from glob import glob

reid_model = ResNet_Loader('model_880_base.ckpt',50)
foldername = raw_input("Folder Name : ")
car_list=[]
for i in glob(foldername+'/*jpg'):
	#name=i.split('/',1)[1]
	car_list.append(i)

features = reid_model.inference(car_list).numpy()
print(features.shape)
 
