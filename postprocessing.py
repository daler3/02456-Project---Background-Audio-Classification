import numpy as np

predictions = []
threshold = 0.5
with open("writepreds.txt", "r") as preds:
	for line in preds:
		predictions.append([1 if float(x) > threshold else 0 for x in line.split(' ')])
		#for x in line.split(' ')[:-1]:
		#	print(float(x) > 0.4)


for prediction in predictions:
	print(prediction)
np.savetxt("postpreds.txt", np.array(predictions))
