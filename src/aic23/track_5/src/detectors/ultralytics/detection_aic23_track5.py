from ultralytics import YOLO

# Load a model
model = YOLO("runs/aic23_track5/yolov8x6_1280_7cls/weights/best.pt")  # load a custom model

# Predict with the model
# results = model(
# 	source="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/"
# 	"Track_5/aicity2023_track5/images/"
# 	"007/00700128.jpg",
# 	conf=0.001,
# 	save=True,  # save plot result
# 	show=True,  # show result
# 	save_txt=True,  # save result in txt
# 	save_conf=True,  # in result has conf score
# 	save_json=True  # save json file result
# )  # predict on an image

results = model(
	source="/media/sugarubuntu/DataSKKU3/3_Dataset/AI_City_Challenge/2023/"
		   "Track_5/aicity2023_track5/videos/"
		   "002.mp4",
	conf=0.001,
	save=True,
	show=True,
)  # predict on a video

print(type(results))
print(results)
