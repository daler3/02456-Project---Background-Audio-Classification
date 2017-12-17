import matplotlib
matplotlib.use('Agg')
from preprocessor import preprocessor
from keras_models import piczak_CNN, piczak_CNN_Salik, piczak_mod_CNN, piczak_mod_CNN2
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn import metrics
from keras.models import load_model
import numpy as np
import pandas as pd
import utils
import sklearn.metrics
from sklearn.metrics import f1_score, hamming_loss, zero_one_loss
import matplotlib.pyplot as plt
from pandas_confusion import ConfusionMatrix
import keras.backend as K
import tensorflow as tf

classes2 = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
			'jackhammer', 'siren', 'street_music',
			'air_conditioner+car_horn', 'air_conditioner+children_playing', 'air_conditioner+dog_bark',
			'air_conditioner+drilling', 'air_conditioner+engine_idling', 'air_conditioner+gun_shot',
			'air_conditioner+jackhammer', 'air_conditioner+siren', 'air_conditioner+street_music',
			'car_horn+children_playing',
			'car_horn+dog_bark', 'car_horn+drilling', 'car_horn+engine_idling', 'car_horn+gun_shot',
			'car_horn+jackhammer',
			'car_horn+siren', 'car_horn+street_music', 'children_playing+dog_bark', 'children_playing+drilling',
			'children_playing+engine_idling', 'children_playing+gun_shot', 'children_playing+jackhammer',
			'children_playing+siren', 'children_playing+street_music', 'dog_bark+drilling', 'dog_bark+engine_idling',
			'dog_bark+gun_shot', 'dog_bark+jackhammer', 'dog_bark+siren', 'dog_bark+street_music',
			'drilling+engine_idling', 'drilling+gun_shot', 'drilling+jackhammer', 'drilling+siren',
			'drilling+street_music', 'engine_idling+gun_shot', 'engine_idling+jackhammer', 'engine_idling+siren',
			'engine_idling+street_music', 'gun_shot+jackhammer', 'gun_shot+siren', 'gun_shot+street_music',
			'jackhammer+siren', 'jackhammer+street_music', 'siren+street_music']

single_classes = ['ac', 'ch', 'cp', 'db', 'd', 'ei', 'gs', 'j', 's', 'sm']
combined_classes = ['ac+ch', 'ac+cp', 'ac+db',
					'ac+d', 'ac+ei', 'ac+gs',
					'ac+j', 'ac+s', 'ac+sm',
					'ch+cp',
					'ch+db', 'ch+d', 'ch+ei', 'ch+gs',
					'ch+j',
					'ch+s', 'ch+sm', 'cp+db', 'cp+d',
					'cp+ei', 'cp+gs', 'cp+j',
					'cp+s', 'cp+sm', 'db+d', 'db+ei',
					'db+gs', 'db+j', 'db+s', 'db+sm',
					'd+ei', 'd+gs', 'd+j', 'd+s',
					'd+sm', 'ei+gs', 'ei+j', 'ei+s',
					'ei+sm', 'gs+j', 'gs+s', 'gs+sm',
					'j+s', 'j+sm', 's+sm']
classes = ['ac', 'ch', 'cp', 'db', 'd', 'ei', 'gs', 'j', 's', 'sm',
		   'ac+ch', 'ac+cp', 'ac+db',
		   'ac+d', 'ac+ei', 'ac+gs',
		   'ac+j', 'ac+s', 'ac+sm',
		   'ch+cp',
		   'ch+db', 'ch+d', 'ch+ei', 'ch+gs',
		   'ch+j',
		   'ch+s', 'ch+sm', 'cp+db', 'cp+d',
		   'cp+ei', 'cp+gs', 'cp+j',
		   'cp+s', 'cp+sm', 'db+d', 'db+ei',
		   'db+gs', 'db+j', 'db+s', 'db+sm',
		   'd+ei', 'd+gs', 'd+j', 'd+s',
		   'd+sm', 'ei+gs', 'ei+j', 'ei+s',
		   'ei+sm', 'gs+j', 'gs+s', 'gs+sm',
		   'j+s', 'j+sm', 's+sm', 'xxx']

dic_tot = {}


def classes_number_mapper():
	dic_single_classes = {}
	# dic_tot = {}
	x = 0
	for s in single_classes:
		dic_single_classes[s] = x + 1
		x += 1
	for n in range(0, 10):
		dic_tot[n + 1] = n
	x = 10
	for s in combined_classes:
		splitted_s = s.split("+")
		t1 = dic_single_classes[splitted_s[0]]
		t2 = dic_single_classes[splitted_s[1]]
		t_comb = int(str(t1) + str(t2))
		dic_tot[t_comb] = x
		x += 1
	dic_tot[1000000] = x


train_dirs = []
save_dir = '../../data/UrbanSound8K/extracted_long_just_overlap'


def one_hot_encode(labels):
	"""
	Convert labels to one-hot matrix. Each row is a label, each column is a unique label.
	:param labels:
	:return: one-hot-vector matrix for the labels
	"""
	n_labels = len(labels)
	# n_unique_labels = len(np.unique(labels))
	n_unique_labels = 56
	one_hot_encode = np.zeros((n_labels, n_unique_labels))
	# print (labels)
	# print (n_labels)
	# print (np.arange(n_labels))
	one_hot_encode[np.arange(n_labels), labels] = 1
	return one_hot_encode


def transform_labels_numbers(labels_s):
	new_labels = []
	for l in labels_s:
		new_labels.append(dic_tot[l])
	return new_labels


def from_plus_to_one_hot(pred_labels):
	final_conv_labels = []
	list_pl = list(pred_labels)
	for l in list_pl:
		l_list = list(l)
		temp_index = [i for i, j in enumerate(l_list) if j == 1]
		if len(temp_index) == 1:
			final_conv_labels.append(temp_index[0] + 1)
		if len(temp_index) == 0:
			final_conv_labels.append(1000000)
		if len(temp_index) > 1:
			t_1 = temp_index[0] + 1
			t_2 = temp_index[1] + 1
	final_conv_labels_1hot = transform_labels_numbers(final_conv_labels)
	# for f in final_conv_labels:
	#	final_conv_labels_1hot.append(f - 1)
	final_conv_labels_1hot = one_hot_encode(final_conv_labels_1hot)
	return final_conv_labels_1hot


def train_keras_cnn(epochs=25, output_model_file="./piczak_model_fold1_only.h5",
					output_predictions_file="./test.csv"):
	pp = preprocessor(parent_dir='../UrbanSound8K/audio')
	print("Loading the data...")

	# Run on all audio files in the 10 folders
	# MULTI LABEL
	# train_dirs = ["audio_overlap/folder1_overlap", "audio_overlap/folder2_overlap", "audio_overlap/folder3_overlap", "audio_overlap/folder4_overlap", "audio_overlap/folder5_overlap", "audio_overlap/folder6_overlap", "audio_overlap/folder7_overlap", "audio_overlap/folder8_overlap", "audio_overlap/folder9_overlap","audio_overlap/folder10_overlap"]

	# SINGLE LABEL
	train_dirs = ["fold1", "fold2", "fold3", "fold4", "fold5", "fold6", "fold7", "fold8", "fold9", "fold10"]
	pp.data_prep(train_dirs=train_dirs, load_path="../UrbanSound8K/audio/extracted_short_60")
	# pp.data_prep(train_dirs=train_dirs, load_path=save_dir)

	tb = TensorBoard(log_dir='./TensorBoard/piczak_CNN_singlelabel')

	print("model creation")
	model = piczak_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])

	print("model fitting")
	model.fit(pp.train_x, pp.train_y, validation_split=.1, epochs=epochs,
			  batch_size=256, verbose=2, callbacks=[tb])

	print("model evaluation")
	scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
	print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

	print("Writing test predictions to csv file : {0}".format(output_predictions_file))

	def write_preds(preds, fname):
		# y_pred.argmax(1)
		pd.DataFrame({"Predictions": preds, "Actual": np.array(pp.test_y).argmax(1)}).to_csv(fname, index=False,
																							 header=True)

	preds = model.predict(pp.test_x)

	np.savetxt("postpreds.txt", np.array(preds))

	# ************ PROCESSING THE PREDICTIONS
	preds[preds >= 0.5] = 1
	preds[preds < 0.5] = 0

	np.savetxt("truelabels.txt", np.array(pp.test_y))
	np.savetxt("writepreds.txt", np.array(preds))
	# write_preds(preds, output_predictions_file)
	print("F1 SCORE:")
	print(f1_score(pp.test_y, preds, average=None))

	print("Hamming Loss:")
	print(hamming_loss(pp.test_y, preds))
	print("Zero-one loss:")
	print(zero_one_loss(pp.test_y, preds))

	# write_preds(preds, output_predictions_file)
	# confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
	# utils.plot_confusion_matrix(confusion_matrix, classes)


	# confusion_matrix = metrics.confusion_matrix(np.argmax(pp.test_y, axis=1), preds)
	# utils.plot_confusion_matrix(confusion_matrix, classes)

	# I reach here in plus_one_hot_encode, I want to transform it in one hot
	y_test_preds = from_plus_to_one_hot(np.array(pp.test_y))
	preds_transf = from_plus_to_one_hot(np.array(preds))

	cm = ConfusionMatrix(np.array(y_test_preds).argmax(1), np.array(preds_transf).argmax(1))
	print(cm)

	plt.figure(figsize=(64, 32))
	ax = cm.plot()

	ax.set_xticklabels(classes, rotation="vertical")
	ax.set_yticklabels(classes)
	plt.savefig("cmpre.png")
	plt.savefig("cm.png", dpi=600)

	model.save("saved/Piczak_CNN_pretrain_single_label.h5")


def evaluateSavedModel():
	model = load_model("saved/Piczak_CNN_pretrainsingle_trainmulti.h5")
	pp = preprocessor(parent_dir='../UrbanSound8K/audio')
	train_dirs = ["audio_overlap/folder1_overlap", "audio_overlap/folder2_overlap", "audio_overlap/folder3_overlap",
				  "audio_overlap/folder4_overlap", "audio_overlap/folder5_overlap", "audio_overlap/folder6_overlap",
				  "audio_overlap/folder7_overlap", "audio_overlap/folder8_overlap", "audio_overlap/folder9_overlap",
				  "audio_overlap/folder10_overlap"]

	#pp.data_prep(train_dirs=[], test_fold="fold2", load_path="../UrbanSound8K/audio/extracted_short_60/")
	pp.data_prep(train_dirs=train_dirs, test_fold="fold2", save_path='../extracted_feat/')

	tb = TensorBoard(log_dir='./TensorBoard/piczak_CNN_singlelabel_pretrain_continue_multilabel')
	# model.fit(pp.train_x, pp.train_y,validation_split=.1, epochs=25,
	#		  batch_size=256, verbose=2, callbacks=[tb])



	print("model evaluation")
	scores = model.evaluate(pp.test_x, pp.test_y, verbose=0)
	print("loss: {0}, test-acc: {1}".format(scores[0], scores[1]))

	preds = model.predict(pp.test_x)

	evaluateModel(pp, preds)


def evaluateModel(pp, preds, fold):
	# ************ PROCESSING THE PREDICTIONS
	preds[preds >= 0.5] = 1
	preds[preds < 0.5] = 0
	print("F1 SCORE:")
	print(f1_score(pp.test_y, preds, average=None))
	print("Hamming Loss:")
	print(hamming_loss(pp.test_y, preds))
	print("Zero-one loss:")
	print(zero_one_loss(pp.test_y, preds))
	# I reach here in plus_one_hot_encode, I want to transform it in one hot
	y_test_preds = from_plus_to_one_hot(np.array(pp.test_y))
	preds_transf = from_plus_to_one_hot(np.array(preds))
	cm = ConfusionMatrix(np.array(y_test_preds).argmax(1), np.array(preds_transf).argmax(1))
	print(cm)
	ax = cm.plot()
	ax.set_xticklabels(classes, rotation="vertical")
	ax.set_yticklabels(classes)
	plt.savefig("cmpre{0}.png".format(fold))



# logging.basicConfig(filename='cv.log', filemode='w', level=logging.DEBUG)


def piczac_cross_validation(epochs, load_path):
	train_dirs = []

	n_folders = 10

	for i in range(1, n_folders + 1):
		#train_dirs.append('fold{0}'.format(i))
		train_dirs.append('folder{0}_overlap'.format(i))

	print(train_dirs)
	for fold in [(9, 10), (2, 5), (6, 7)]:
		val_fold = 'folder{0}_overlap'.format(fold[0])
		test_fold = 'folder{0}_overlap'.format(fold[1])
		#val_fold = 'fold{0}'.format(fold[0])
		#test_fold = 'fold{0}'.format(fold[1])
		train_dirs.remove(val_fold)
		train_dirs.remove(test_fold)

		pp = preprocessor(parent_dir='../../data/UrbanSound8K/audio')
		pp.data_prep(train_dirs=train_dirs, val_fold=val_fold, test_fold=test_fold, load_path=load_path)

		model = piczak_CNN(input_dim=pp.train_x[0].shape, output_dim=pp.train_y.shape[1])
		print ("done")
		print("OPTIMIZER")
		#print(model.optimizer.lr)
		#K.set_value(model.optimizer.lr, 0.002)
		#model.optimizer.lr.set_value(0.0001)
		#model.save('Models/model1_all_p2_bn{0}.h5'.format(str(fold)))
		#model = load_model('Models/model1_all_p2{0}.h5'.format(str(fold)))
		#model = load_model('Models/model1_all_p2_bnsec_overlap_{0}.h5'.format(str(fold)))




		tb = TensorBoard(log_dir='./TensorBoard/' + 'overlap_run{0}'.format(fold[1]))
		es = EarlyStopping(patience=10, verbose=1)

		model.fit(pp.train_x, pp.train_y, validation_data=[pp.val_x, pp.val_y], epochs=epochs,
				  batch_size=1000, verbose=2, callbacks=[tb, es])
		#model.save('Models/model1_all_p2_bnsec_overlap_9010_{0}.h5'.format(str(fold)))

		preds = model.predict(pp.test_x)
		evaluateModel(pp,preds, fold)

		K.clear_session()

		train_dirs.append(val_fold)
		train_dirs.append(test_fold)
	# val_fold = 'fold' + str(folds[0])
	# test_fold= 'fold' + str(folds[1])


if __name__ == '__main__':
	K.clear_session()
	tf.reset_default_graph()
	print("main")
	classes_number_mapper()
	print(dic_tot)
	for i in sorted(dic_tot.keys()):
		print(i, dic_tot[i])
	# train_keras_cnn()
	#evaluateSavedModel()
	piczac_cross_validation(125, "../UrbanSound8K/audio/extracted_overlapping_50/audio_overlap")
	#piczac_cross_validation(125, "../../feat_overlap_diff")