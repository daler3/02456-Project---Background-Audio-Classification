from preprocessor import preprocessor
from keras_models import piczak_CNN
from keras.callbacks import TensorBoard
from sklearn import metrics
import numpy as np
import pandas as pd
import utils

classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
train_dirs = []
save_dir = '../UrbanSound8K/extracted_long'
def train_keras_cnn(epochs=25, output_model_file="./piczak_model_fold1_only.h5",
					output_predictions_file="./test.csv"):

	train_dirs = ['fold1']
	pp = preprocessor(parent_dir='../UrbanSound8K/audio')
	print("Loading the data...")
	# Run this to load the extracted features
	# pp.data_prep(train_dirs=["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"], load_path='extracted')
	#pp.data_prep(train_dirs=train_dirs, load_path=save_dir)

	tb = TensorBoard(log_dir='./TensorBoard')
	#print("fold " + str(i) + " tried in fold")
	#train_dirs.remove('fold{0}'.format(i))
	#test_dir='fold{0}'.format(i)
	#print("Run {0}: test folder is fold{0}".format(i))
	print("data prep")
	pp.data_prep(train_dirs=train_dirs)#, load_path='../UrbanSound8K/extracted_long')
	#pp.data_prep(train_dirs=train_dirs, load_path='../UrbanSound8K/extracted_long', test_fold=test_dir)
	#train_dirs.append('fold{0}'.format(i))
	print("model creation")
	model = piczak_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])

	print("model fitting")
	model.fit(pp.train_x, pp.train_y, epochs=epochs,
			  batch_size=256, verbose=2, callbacks=[tb])

	#print("Saving the model to file: {0}".format(output_model_file))
	#model.save(output_model_file)
	print("model evaluation")
	scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
	print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

	#print("Writing test predictions to csv file : {0}".format(output_predictions_file))
	#def write_preds(preds, fname):
	#    pd.DataFrame({"Predictions": preds, "Actual": np.argmax(pp.test_y, axis=1)}).to_csv(fname, index=False,
	#                                                                                      header=True)

	#preds = model.predict_classes(pp.test_x, verbose=0)
	#write_preds(preds, output_predictions_file)
	#confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
	#utils.plot_confusion_matrix(confusion_matrix, classes)




if __name__ == '__main__':
	print("main")
	train_keras_cnn()
