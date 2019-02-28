from ReID_CNN.Model_Wrapper import ResNet_Loader
from glob import glob

reid_model = ResNet_Loader('model_880_base.ckpt',50)

car_list=[]
for i in glob('frame_set_extracted/*jpg'):
	#name=i.split('/',1)[1]
	car_list.append(i)

features = reid_model.inference(car_list).numpy()
print(features.shape)
 
